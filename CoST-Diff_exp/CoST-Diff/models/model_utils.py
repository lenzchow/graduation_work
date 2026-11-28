import torch
import math
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import os
from common.spatial_func import distance, cal_loc_along_line, SPoint

from utils.utils import load_json_data
import numpy as np
import torch
from collections import Counter

# kl_loso计算
def freq_kl_loss(pred_prob, global_bs_freq):
    """
    pred_prob: [B, T, N] 模型预测的每步基站分布（概率）
    global_bs_freq: [N] 全局基站分布
    """
    # 先求平均预测分布
    avg_pred = pred_prob.mean(dim=(0,1))  # [N]

    # 对数避免 nan
    avg_pred = torch.clamp(avg_pred, 1e-8, 1.0)
    global_bs_freq = torch.clamp(global_bs_freq, 1e-8, 1.0)

    kl = F.kl_div(avg_pred.log(), global_bs_freq, reduction='batchmean')
    return kl
########################################################################

###### 优化器

########################################################################
def freeze_and_rebuild_optimizer(model, base_lr):
    print("🔒 Freezing diffusion backbone & rebuilding optimizer")

    # 1. 冻结 diffusion 相关模块
    for name, p in model.named_parameters():
        if (
            name.startswith("diff_model")
            or name.startswith("time_mlp")
            or name.startswith("mask_step_embed")
        ):
            p.requires_grad = False

    # 2. 只收集仍然可训练的参数
    trainable_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_params.append(p)

    # 3. 新 optimizer（单 lr，稳）
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=base_lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5
    )

    return optimizer, scheduler

########################################################################
def compute_global_bs_frequency(all_index_seqs, num_bs):
    """
    all_index_seqs: dict[int, list[list[int]]]
    num_bs: 基站总数（embedding 数量）

    return:
        freq: torch.Tensor [num_bs]，归一化访问频率
        count: torch.Tensor [num_bs]，原始计数
    """
    counter = Counter()

    for traj_len, traj_list in all_index_seqs.items():
        for traj in traj_list:
            counter.update(traj)

    count = torch.zeros(num_bs, dtype=torch.float32)

    for bs_id, c in counter.items():
        count[bs_id] = c

    freq = count / count.sum()

    return freq, count
#####################################################################################################
#
# 起始点和起始时间采样函数
#
#####################################################################################################
class ProtoHead(nn.Module):
    def __init__(self, in_dim, n_class, init_proto=None, proj_dim=None, tau_init=20.0):
        """
        in_dim   : 输入特征维度（这里 = 轨迹空间通道维度 Ds）
        n_class  : 类别数（基站数 |V|）
        init_proto: 可选，用已有 embedding 初始化原型
        proj_dim : 可选，先做一个线性投影到更小维度
        tau_init : 初始温度（越大越“尖锐”）
        """
        super().__init__()
        self.proj = (nn.Linear(in_dim, proj_dim, bias=False)
                     if proj_dim else nn.Identity())
        D = proj_dim or in_dim

        self.proto = nn.Parameter(torch.randn(n_class, D))
        self.log_s = nn.Parameter(torch.log(torch.tensor(float(tau_init))))
        nn.init.normal_(self.proto, std=0.02)

        if init_proto is not None:
            with torch.no_grad():
                P = init_proto
                if proj_dim is not None and P.size(1) != proj_dim:
                    # 如果维度对不上，可以先线性变换到 proj_dim 再赋值
                    P = F.normalize(P, dim=-1)
                    self.proto.copy_(P)
                elif P.size(1) == D:
                    self.proto.copy_(P)

    def forward(self, x):
        """
            x: [..., D_in]
            return: [..., n_class] logits
        """
        # h = self.proj(x)                     # [..., D]
        h = F.normalize(x, dim=-1)
        proto = F.normalize(self.proto, dim=-1)  # [V, D]
        logits = torch.einsum('...d,vd->...v', h, proto)
        logits = logits * self.log_s.exp()       # 温度缩放
        return logits


#####################################################################################################
#
# 起始点和起始时间采样函数
#
#####################################################################################################
import numpy as np
import torch

@torch.no_grad()
def sample_cond_for_generation(cond_feat: torch.Tensor,
                               all_index_dict: dict,
                               all_tim_dict: dict,
                               all_slot_seqs: dict, 
                               all_seq_cond: dict,
                               length: int,
                               batchsize: int,
                               device: torch.device,
                               tim_unit: str = "auto",
                               include_city: bool = False,
                               city_col_idx: int = 4):
    """
    cond_feat 列: [lng_z, lat_z, ac_id, district_id, (可选 city_id)]

    all_index_dict[length] : List[np.ndarray] or List[List[int]]  (轨迹的基站 index 序列)
    all_tim_dict[length]   : List[np.ndarray] or List[List[float]] (对应的时间序列)
    all_slot_seqs[length]  : List[List[int]]  (事件级条件：时间段标签，长度=length)
    all_seq_cond[length]   : List[int]        (序列级条件：weekday 0..6)

    返回 dict:
      {
        'ac_id'   : LongTensor[B],
        'dist_id' : LongTensor[B],
        'lnglat'  : FloatTensor[B, 2],
        'minute'  : LongTensor[B],       # 起点分钟数
        'bs0'     : LongTensor[B],       # 起点基站ID
        'city_id' : LongTensor[B] or None,
        'weekday' : LongTensor[B],       # 序列级条件
        'slot'    : LongTensor[B, L],    # 事件级条件：时间段标签序列
      }
    """
    assert length in all_index_dict and length in all_tim_dict,  f"no bucket L={length} in index/tim"
    assert length in all_slot_seqs and length in all_seq_cond,   f"no bucket L={length} in slot/seq_cond"

    real_bs_list   = all_index_dict[length]   # list[...]，每条 shape=(L,)
    real_tim_list  = all_tim_dict[length]     # list[...]，每条 shape=(L,)
    real_slot_list = all_slot_seqs[length]    # list[List[int]]
    real_seq_list  = all_seq_cond[length]     # list[int]

    n_real = len(real_bs_list)
    assert n_real > 0, f"empty bucket L={length}"
    assert len(real_tim_list)  == n_real
    assert len(real_slot_list) == n_real
    assert len(real_seq_list)  == n_real

    # 在该长度 bucket 内随机采样 batchsize 条轨迹作为条件来源
    sel = np.random.choice(n_real, size=batchsize, replace=(batchsize > n_real))

    # 起点基站
    bs0 = torch.as_tensor([real_bs_list[i][0] for i in sel],
                          device=device).long()      # [B]

    # 起始时间 → 分钟（和训练时同口径）
    t0  = torch.as_tensor([real_tim_list[i][0] for i in sel],
                          device=device, dtype=torch.float32)  # [B]
    if tim_unit == "auto":
        tim_unit = "sec" if float(t0.max()) > 1500 else "min"
    if tim_unit == "sec":
        minute = (((t0 % 86400.0) / 60.0).floor()).long()
    else:
        minute = torch.remainder(t0, 1440.0).floor().long()     # [B]

    # 静态基站特征
    cond_feat = cond_feat.to(device)
    N = cond_feat.size(0)
    assert int(bs0.min()) >= 0 and int(bs0.max()) < N, f"bs0 out of range: max={int(bs0.max())} N={N}"
    f = cond_feat.index_select(0, bs0)  # [B, F]

    lnglat  = f[:, :2].to(torch.float32).contiguous()  # [B,2]
    ac_id   = f[:, 2].long()                           # [B]
    dist_id = f[:, 3].long()                           # [B]

    city_id = None
    if include_city:
        assert f.size(1) > city_col_idx, "cond_feat 没有 city 列"
        city_id = f[:, city_col_idx].long()            # [B]

    # === 序列级条件：weekday (0..6)，一条轨迹一个值 ===
    weekday = torch.as_tensor([real_seq_list[i] for i in sel],
                              device=device).long()    # [B]

    # === 事件级条件：时间段标签序列 slot_seq (0/1/2)，长度=L ===
    slot_seqs = [real_slot_list[i] for i in sel]       # list[List[int]]，每条长度=length
    slot_seqs = torch.as_tensor(slot_seqs,
                                device=device).long()  # [B, L]

    return {
        'ac_id':   ac_id,
        'dist_id': dist_id,
        'lnglat':  lnglat,
        'minute':  minute,
        'bs0':     bs0,
        'city_id': city_id,
        'weekday': weekday,   # 序列级条件
        'slot':    slot_seqs, # 事件级条件
        'source_index': torch.as_tensor(sel, device=device).long()
    }

#####################################################################################################
#
# 时间极大似然计算
#
#####################################################################################################
def moln_tail_nll(
    pi, mu, log_sigma, dt,
    tail_alpha=2.0,
    tail_threshold=100.0
):
    """
    pi:        [B,T,K]  (softmax 后)
    mu:        [B,T,K]
    log_sigma: [B,T,K]
    dt:        [B,T]
    """

    # ---------- 数值安全 ----------
    dt = dt.clamp(min=1e-2, max=2000.0)
    log_dt = torch.log(dt).unsqueeze(-1)

    log_sigma = log_sigma.clamp(-3.0, 2.0)
    sigma = torch.exp(log_sigma)

    # ---------- log p(dt | k) ----------
    log_prob = (
        -0.5 * ((log_dt - mu) / sigma) ** 2
        - log_sigma
        - 0.5 * np.log(2 * np.pi)
    )

    # ---------- mixture ----------
    pi = pi.clamp(min=1e-6)
    log_mix = torch.log(pi) + log_prob
    log_likelihood = torch.logsumexp(log_mix, dim=-1)

    # ---------- smooth tail reweight ----------
    # soft tail: weight grows with dt
    tail_weight = 1.0 + tail_alpha * torch.sigmoid(
        (dt - tail_threshold) / 20.0
    )

    nll = -log_likelihood * tail_weight

    if not torch.isfinite(nll).all():
        print("🔥 NLL forward NaN")
        print("dt:", dt.min(), dt.max())
        print("mu:", mu.min(), mu.max())
        print("log_sigma:", log_sigma.min(), log_sigma.max())
        print("pi:", pi.min(), pi.max())
        print("tail_weight:", tail_weight.min(), tail_weight.max())
        raise RuntimeError("NLL forward NaN")
    
    return nll.mean()


#########################################################################################…209 tokens truncated…获取起始时间分布 ===
    initial_prob = model.dataset_tim_prob  # 长度 = tim_size // 10

    # === 2. 采样起始时间 ===
    start_times = np.random.choice(
        np.arange(tim_size // 10), 
        size=ntrajs, 
        p=initial_prob
    ) * 10 + 5  # 每个时间段的中心点，shape: (ntrajs,)

    # === 3. 累加等待时间生成到达时间 ===
    arrival_times = []
    for i in range(ntrajs):
        wait_times = wait_times_list[i]  # 当前轨迹的等待时间序列
        traj_times = [start_times[i]]    # 第一个时间为起始时间
        for w in wait_times:
            traj_times.append(traj_times[-1] + w)
        arrival_times.append(traj_times)

    return torch.tensor(arrival_times, dtype=torch.float32, device=model.device)


#####################################################################################################
#
# 时间采样函数
#
#####################################################################################################
@torch.no_grad()
def sample_moln(pi, mu, log_sigma):
    """
    pi: [B,T,K]
    mu, log_sigma: same
    """
    B, T, K = pi.shape

    comp = torch.multinomial(
        pi.view(-1, K), 1
    ).view(B, T)

    mu_sel = mu.gather(-1, comp.unsqueeze(-1)).squeeze(-1)
    log_sig_sel = log_sigma.gather(-1, comp.unsqueeze(-1)).squeeze(-1)

    eps = torch.randn_like(mu_sel)
    log_dt = mu_sel + torch.exp(log_sig_sel) * eps
    return torch.exp(log_dt)


#####################################################################################################
#
# 时间分布过滤器
#
#####################################################################################################
def timprob_filter(tout, mask=None, training=True):
    """
    对时间分布参数进行过滤或调整
    参数:
        tout: torch.Tensor, log λ，形状 [batch_size, seq_len]
        mask: torch.Tensor, 同尺寸掩码，1表示有效时间区间，0表示无效
        training: bool, 训练或推理模式
    返回:
        过滤后的 log λ
    """
    if mask is not None:
        if training:
            tout = tout - torch.log(mask + 1e-10)  # 对无效区间加惩罚
        else:
            tout = tout.masked_fill(mask != 1, float('-inf'))
    return tout


def timprob_filter(tout, valid_range=None, training=True):
    """
    对时间分布参数 (log λ) 进行过滤或调整
    参数:
        tout: torch.Tensor, log λ，形状 [batch_size, seq_len]
        valid_range: tuple(lower, upper)，允许的时间范围（单位与停留时间一致）
        training: bool, 训练或推理模式
    返回:
        过滤后的 log λ (torch.Tensor)
    """
    if valid_range is not None:
        lower, upper = valid_range

        # 构造掩码 (mask): 在合法区间为1，否则为0
        # 这里假设 tout 尚未采样，所以只做soft约束/屏蔽
        mask = ((torch.exp(tout) >= 1.0/upper) & (torch.exp(tout) <= 1.0/lower)).float()

        if training:
            # 训练时：对非法区域的 λ 增加惩罚（降低概率）
            tout = tout - torch.log(mask + 1e-10)  
        else:
            # 推理时：直接屏蔽非法区域
            tout = tout.masked_fill(mask != 1, float('-inf'))

    return tout


def sample_time_filtered(tout, valid_range=(1, 120), infer_maxlast=1440, return_tensor=True, device=None):
    """
    结合 timprob_filter 的采样时间函数
    """
    # 1. 过滤时间分布
    tout_filtered = timprob_filter(tout, valid_range=valid_range, training=False)

    # 2. 采样 Δt
    return sample_time(tout_filtered, infer_maxlast=infer_maxlast, return_tensor=return_tensor, device=device)

#####################################################################################################
#
# 时间傅立叶编码函数
#
#####################################################################################################
import numpy as np
import torch

def abs_tim_emb_forward(x, d_model, device, fourier: bool):
    """
    x: [B, T]，这里我们用来编码 Δt 或 t
    fourier=True  → 使用论文 (6) 里类似的 Fourier PE
    fourier=False → 使用 Transformer 风格的 1e-4^{2i/d}
    """
    x = torch.as_tensor(x, dtype=torch.float32, device=device)  # 强制成 GPU float32
    
    # i = 0,1,...,d_model-1 ；a = floor(i/2) * 2/d
    a = torch.div(torch.arange(0.0, d_model), 2, rounding_mode='floor').to(device)
    a = a * 2.0 / d_model   # 归一化频率

    x = x.to(device)        # [B, T]
    x = x.unsqueeze(-1)     # [B, T, 1]

    if fourier:
        # Fourier-based PE: sin(2π i t / γ), cos(2π i t / γ)
        # 这里 γ 取 1440（一天的分钟数）
        freq = (2.0 * np.pi / 1440.0) * a      # [d_model]
        b = torch.matmul(x, freq.view(1, 1, -1))   # [B, T, d_model]
    else:
        # 经典 Transformer 风格：1e-4^{2i/d}
        freq = (1e-4 ** a)                      # [d_model]
        b = torch.matmul(x, freq.view(1, 1, -1))   # [B, T, d_model]

    # 交替填充 sin / cos
    c = torch.zeros_like(b)
    c[:, :, 0::2] = b[:, :, 0::2].sin()
    c[:, :, 1::2] = b[:, :, 1::2].cos()
    return c    # [B, T, d_model]

def encode_time_list(tim_data_list, d_model=128, fourier=False):
    """
    对时间数据列表进行傅里叶编码的完整流程函数
    :param tim_data_list: 时间数据列表，shape为 [batch_size, seq_len]
    :param d_model: 嵌入维度，默认128
    :param fourier: 控制傅里叶编码模式的布尔值，默认False
    :return: 编码结果numpy数组，shape为 [batch_size, seq_len, d_model]
    """
    # 确定计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将二维列表转为tensor并放到对应设备
    time_tensor = torch.tensor(tim_data_list, dtype=torch.float32).to(device)
    
    # 调用编码计算函数
    encoded = abs_tim_emb_forward(time_tensor, d_model, device, fourier)
    
    # 转成numpy数组，detach和cpu是为了从计算图分离并放到CPU
    return encoded
#####################################################################################################
#
# 时间分布统计函数
#
#####################################################################################################
@torch.no_grad()
def sample_arrival_times(model, base_station_seqs, wait_times_list, is_holiday=None,
                         tim_size=1440, bin_size=10):
    """Sample start times by weekday/weekend category, then add generated waits."""
    waits = torch.as_tensor(wait_times_list, dtype=torch.float32, device=model.device)
    if waits.ndim != 2:
        raise ValueError(f"wait_times_list must be 2-D, got shape {tuple(waits.shape)}")
    batch_size = waits.shape[0]
    if is_holiday is None:
        holiday_flags = np.zeros(batch_size, dtype=np.int64)
    else:
        holiday_flags = torch.as_tensor(is_holiday).detach().cpu().numpy().reshape(-1)
        if len(holiday_flags) != batch_size:
            raise ValueError("is_holiday must contain one flag per trajectory")
    distributions = model.dataset_tim_prob
    bins = tim_size // bin_size
    start_minutes = []
    for flag in holiday_flags:
        key = "holiday" if int(flag) else "workday"
        probabilities = np.asarray(distributions[key], dtype=float)
        probabilities = probabilities / probabilities.sum()
        start_bin = np.random.choice(bins, p=probabilities)
        start_minutes.append(start_bin * bin_size + bin_size / 2)
    start = torch.as_tensor(start_minutes, dtype=waits.dtype, device=waits.device).unsqueeze(1)
    return torch.cat([start, start + torch.cumsum(waits, dim=1)], dim=1)


def inference_initial_from_tim_dict(tim_data_dict, seq_cond_dict, tim_size=1440, bin_size=10):
    """Estimate start-time distributions for weekdays (0-4) and weekends (5-6)."""
    grouped = {"workday": [], "holiday": []}
    for length, trajectories in tim_data_dict.items():
        weekdays = seq_cond_dict.get(length, [])
        if len(trajectories) != len(weekdays):
            raise ValueError(f"Time/weekday counts differ for trajectory length {length}")
        for trajectory, weekday in zip(trajectories, weekdays):
            if trajectory:
                key = "holiday" if int(weekday) >= 5 else "workday"
                grouped[key].append(float(trajectory[0]) % tim_size)
    pooled = grouped["workday"] + grouped["holiday"]
    if not pooled:
        raise ValueError("No start timestamps available to estimate time distributions")
    result = {}
    bins = tim_size // bin_size
    for key in ("workday", "holiday"):
        values = grouped[key] or pooled
        counts = np.histogram(values, bins=bins, range=(0, tim_size))[0].astype(float)
        result[key] = counts / counts.sum()
    return result

#####################################################################################################
#
# 基站生成函数
#
#####################################################################################################
def touli(index_uli_dict, 
          index_lat_lon_dict, 
          index_areacode_dict, 
          times, stas, rids, 
          parameters, 
          min_len, max_len, 
          formatted_time, 
          save_txt_num = 0 ):

    rids = rids.permute(1, 0)
    times = times.permute(1, 0)
    stas = stas.permute(1, 0)

    batch_size = rids.size(1)
    trg_len = rids.size(0)
    seqs = torch.zeros(trg_len, batch_size, 2).to("cuda:0")
    traj_save_path = './generate_data/{}_{}_{}_diffT{}_trajlen{}-{}_curTime{}/'.format(
        parameters.dataset,
        parameters.module_type,
        parameters.embed_mode,
        parameters.diff_T,
        min_len,
        max_len,
        formatted_time)
    if not os.path.exists(traj_save_path):
        os.makedirs(traj_save_path)
    with open(traj_save_path + str(save_txt_num) +".txt", 'a+') as f:
        for bs in range(batch_size):
            f.write('#, trajectory num :{}\n'.format(bs))
            
            for i in range(trg_len):
                if i == len(stas): # 把等待时间也写进文件中，最后一位补0
                    sta = 0
                else:
                    sta = stas[i][bs]
                time = times[i][bs]
                rid = rids[i][bs]
                uli = index_uli_dict.get(str(rid.item()),rid)
                areacode = index_areacode_dict.get(str(rid.item()),rid)
                lat, lon = index_lat_lon_dict.get(str(rid.item()),rid)[0], index_lat_lon_dict.get(str(rid.item()),rid)[1]
                if rid >= parameters.id_size: continue #parameters.id_size:continue
                
                rid_value = int(rid.item())
                time_value = float(time.item())
                sta_value = float(sta.item()) if torch.is_tensor(sta) else float(sta)
                f.write(', '.join(map(str, [areacode, lat, lon, rid_value, time_value, sta_value])) + chr(10))
                if sta_value <= 0:
                    break
            # print("----------------")
    return seqs


#####################################################################################################
#
#时间解码器 
#
#####################################################################################################


def _stat(name, t):
    print(f"[{name}] min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f} std={t.std().item():.4f}")
        
        
# ---- 强度映射头：raw∈R → tout=logλ ∈ [log(1/U), log(1/t_min)] ----
class LogLambdaHead(nn.Module):
    def __init__(self, t_min: float = 1.0, U: float = 360.0, eps: float = 1e-8, margin: float = 0.03):
        super().__init__()
        self.eps = eps
        self.margin = margin  # 0~0.49
        self.register_buffer('low',  torch.tensor(math.log(1.0/max(U, eps)), dtype=torch.float32))
        self.register_buffer('high', torch.tensor(math.log(1.0/max(t_min, eps)), dtype=torch.float32))

    @torch.no_grad()
    def set_bounds(self, t_min: float, U: float):
        self.low.copy_( torch.tensor(math.log(1.0/max(U, self.eps)),    dtype=self.low.dtype,  device=self.low.device) )
        self.high.copy_(torch.tensor(math.log(1.0/max(t_min, self.eps)), dtype=self.high.dtype, device=self.high.device))

    @torch.no_grad()
    def set_margin(self, margin: float):
        self.margin = float(max(0.0, min(0.49, margin)))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        # raw: [..., 1]
        p = torch.sigmoid(raw)
        
        
        # _stat("sigmoid(p)", p)
        
        
        if self.margin > 0:
            m = self.margin
            p = m + (1.0 - 2.0*m) * p  # 边界留白，防止贴边饱和
        return self.low + (self.high - self.low) * p

class TimeDecoder(nn.Module):
    def __init__(self, in_dim, hid, K=3, num_layers=2):
        super().__init__()
        self.K = K

        self.ln = nn.LayerNorm(in_dim, eps=1e-6)

        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=in_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )

        self.pi_head  = nn.Linear(hid, K)
        self.mu_head  = nn.Linear(hid, K)
        self.sig_head = nn.Linear(hid, K)

        self.min_log_sigma = -4.0
        self.max_log_sigma =  3.0

    def forward(self, x):
        # x: [B, T, D]
        h, _ = self.rnn(self.ln(x))        # [B,T,D]
        h = self.backbone(h)               # [B,T,H]

        logit_pi = self.pi_head(h)
        mu       = self.mu_head(h)
        log_sig  = self.sig_head(h).clamp(
            self.min_log_sigma, self.max_log_sigma
        )

        pi = torch.softmax(logit_pi, dim=-1)
        return pi, mu, log_sig


    
class SpatialDecoder(nn.Module):
    def __init__(self, d_traj, d_time, d_hidden, d_out):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_traj + d_time, d_hidden),
            nn.ReLU(),

            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),

            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),

            nn.Linear(d_hidden, d_out)  # ★ 必须 = bs_embed dim
        )

    def forward(self, z_traj, z_time, bs_embed):
        """
        z_traj:   [B, T, d_traj]
        z_time:   [B, T, d_time]
        bs_embed: [N, d_out]
        """
        h = self.net(torch.cat([z_traj, z_time], dim=-1))  # [B,T,d_out]
        logits = torch.einsum('btf,nf->btn', h, bs_embed)
        return logits



#####################################################################################################
#
# others
#
#####################################################################################################
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self