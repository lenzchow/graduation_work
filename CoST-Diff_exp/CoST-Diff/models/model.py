import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from copy import deepcopy
import numpy as np
from models.diff_module import diff_CSDI
from models.diff_util import diff_forward_x0_constraint, cal_x0_from_noise_ddpm, ProtoHead
from models.model_utils import encode_time_list, sample_cond_for_generation
import math, torch, torch.nn as nn, torch.nn.functional as F
from models.model_utils import TimeDecoder, SpatialDecoder , abs_tim_emb_forward
from models.cond_film import ConditionManager, SelfCondFiLM

class Diff_RNTraj(nn.Module):
    def __init__(self, diff_model, diffusion_hyperparams, args,
                 n_ac, n_dist, device):
        super().__init__()

        self.diff_model = diff_model
        self.diffusion_param = diffusion_hyperparams
        self.embedding_hour_dim = args.embedding_hour_dim
        self.pre_trained_dim = args.pre_trained_dim
        self.use_fourier = args.use_fourier
        self.device = device

        # ===== 条件管理器（★新）=====
        self.cond_mgr = ConditionManager(
            n_ac=n_ac,
            n_dist=n_dist,
            H=self.diff_model.channels,
            d=32
        )

        # ===== 时间解码器 =====
        self.tim_decoder = TimeDecoder(
            in_dim=self.embedding_hour_dim + self.pre_trained_dim,
            hid=args.tim_dec_linear_hid1,
            # t_min=t_min, U=U_head
        )


        self.traj_decoder = SpatialDecoder(
            d_traj = self.pre_trained_dim ,
            d_time = self.embedding_hour_dim, 
            d_hidden = 512,
            d_out=self.embedding_hour_dim,
        )

        self.selfcond_film = SelfCondFiLM(
            in_dim=self.embedding_hour_dim + self.pre_trained_dim,
            H=self.diff_model.channels
        )  

    def forward(self,
            spatial_A_trans,
            BS_Embedding,
            sta_seqs,
            base_station_seqs,
            src_time_seqs,
            cond_feat,
            src_slot_seqs,
            src_seq_cond, 
            real_freq):
        """
        src_slot_seqs : [B, T]   事件级条件：时间段标签(0=平峰,1=早高峰,2=晚高峰)
        src_seq_cond  : [B, 1]   序列级条件：weekday(0..6)
        """

        # ===== 起点信息 =====
        bs0 = base_station_seqs[:, 0].long()
        f   = cond_feat[bs0]

        B, T = base_station_seqs.shape

        start_lnglat  = f[:, :2]
        start_ac_id   = f[:, 2].long()
        start_dist_id = f[:, 3].long()
        start_tim     = src_time_seqs[:, 0]

        # shpae torch.Size([128]) torch.Size([128]) torch.Size([128, 2]) torch.Size([128, 1]) torch.Size([128, 29]) torch.Size([128])
        # print('shpae',start_ac_id.shape,start_dist_id.shape,start_lnglat.shape,src_seq_cond.shape,src_slot_seqs.shape,start_tim.shape)
        # ===== 条件统一生成 =====
        cond = self.cond_mgr(
            ac_id=start_ac_id,
            dist_id=start_dist_id,
            lnglat=start_lnglat,
            weekday=src_seq_cond,
            slot_ids=src_slot_seqs,
            t0_min=start_tim,
            L=T
        )

        # 8) 融合：静态 + 事件级条件
         # ===== backbone 用的条件（不含 event）=====
        cond_global = cond["global"] + cond["tod"]
        cond_event = cond["event"]

        # print('cond["global"].shape,cond["tod"].shape,cond["event"].shape,',cond["global"].shape,cond["tod"].shape,cond["event"].shape)
    
        # 3) 空间 token 嵌入：基站 ID → 向量
        id_embed = BS_Embedding[base_station_seqs]    # [B, T, D_id]
    
        # 4) 时间间隔（sta）嵌入
        if self.use_fourier:
            # 直接对 Δt 做 Fourier 编码
            sta_emb = abs_tim_emb_forward(
                sta_seqs,
                d_model=self.embedding_hour_dim,
                device=BS_Embedding.device,
                fourier=True          # True 表示走 Fourier 分支
            )                         # [B, T-1, D_t]
        else:
            # 原来的 log1p + 小 MLP
            y = torch.log1p(sta_seqs)               # [B, T-1]
            sta_emb = self.time_mlp(y.unsqueeze(-1))  # [B, T-1, D_t]

            
        pad = torch.zeros((sta_emb.size(0), 1, sta_emb.size(2)),
                          device=sta_emb.device)
        sta_emb = torch.cat([sta_emb, pad], dim=1)    # [B, T, D_t]
    
        # 5) 主输入特征：时间间隔 embedding + 基站 embedding
        x = torch.cat([sta_emb, id_embed], dim=-1)    # [B, T, D_t + D_id]
    
        # 9) 送入扩散模型（保持其它参数不变）
        diff_noise, const_loss, x0_loss, sta_loss, freq_kl = diff_forward_x0_constraint(
            net=self.diff_model,
            tim_decoder=self.tim_decoder,      # 时间 TPP 头，返回 tout = logλ
            traj_decoder=self.traj_decoder,
            sta_seqs=sta_seqs,
            base_station_seqs=base_station_seqs,
            x=x,
            diffusion_hyperparams=self.diffusion_param,
            BS_Embedding=BS_Embedding,
            spatial_A_trans=spatial_A_trans,
            real_freq=real_freq,
            embedding_hour_dim=self.embedding_hour_dim,
            sc_film=self.selfcond_film,
            # real_freq = real_freq,
            # ★ 把“序列级 + 事件级”合成后的条件喂给去噪网络
            cond_global=cond_global,
            cond_event=cond_event,
        )
    
        return diff_noise, const_loss, x0_loss, sta_loss, freq_kl

    @torch.no_grad()
    def generate_data(self, spatial_A_trans, SE, batchsize, length,
                  pre_dim, t_min, U_business,
                  cond_feat,
                  all_index_dict, all_tim_dict,
                  all_slot_seqs, all_seq_cond):

        # 1) 采样一批条件（基站 + 时间 + weekday + 时间段标签）
        cond_raw = sample_cond_for_generation(
            cond_feat=cond_feat,
            all_index_dict=all_index_dict,
            all_tim_dict=all_tim_dict,
            all_slot_seqs=all_slot_seqs,
            all_seq_cond=all_seq_cond,
            length=length,
            batchsize=batchsize,
            device=SE.device,
            include_city=False  # 需要 city 时改 True
        )

        # 2) 静态 + 序列级条件编码：StaticCond 输出 [B, T, H]
        cond = self.cond_mgr(
            ac_id=cond_raw["ac_id"],            # [B]
            dist_id=cond_raw["dist_id"],        # [B]
            lnglat=cond_raw["lnglat"],          # [B,2]
            weekday=cond_raw["weekday"],        # [B]
            slot_ids=cond_raw["slot"],          # [B,T]
            t0_min=cond_raw["minute"],          # [B]
            L=length,
        )
    
        # 4) 合并：静态条件 + 事件级条件
        cond_global = cond["global"]             # [B, T, H]
        cond_event = cond['event'] + cond["tod"],
        
        # 5) 扩散反推，生成 x0_tim (logλ) 和 x0_road
        spatial_A_trans = torch.as_tensor(
            spatial_A_trans, dtype=torch.float32, device=SE.device
        )
    
        dt, x0_road = cal_x0_from_noise_ddpm(
            net=self.diff_model,
            tim_decoder=self.tim_decoder,
            traj_decoder=self.traj_decoder,
            diffusion_hyperparams=self.diffusion_param,
            batchsize=batchsize,
            length=length,
            pre_dim=pre_dim,
            tim_embedding_dim=self.embedding_hour_dim,
            BS_Embedding=SE,               # 明确指定 BS_Embedding
            cond_global=cond_global,
            cond_event=cond_event,
        )
    
        return dt, x0_road