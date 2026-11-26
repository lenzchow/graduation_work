import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from copy import deepcopy
import copy
from models.diff_module import diff_CSDI
from models.diff_util import diff_forward_x0_constraint, cal_x0_from_noise_ddpm

class Diff_RNTraj(nn.Module):
    def __init__(self, diff_model, diffusion_hyperparams, args_dict):
        super(Diff_RNTraj, self).__init__()
        self.diff_model = diff_model
        self.diffusion_param = diffusion_hyperparams
        self.tim_emb_size = 

        # 可学习的时间嵌入矩阵（0~23小时）
        # self.time_hour_embedding = nn.Embedding(time_granularity, embedding_hour_dim)
        
        self.emb_tim = nn.Linear(1, self.tim_emb_size, bias=False)
        
        # 嵌入学习矩阵维度： 32
        self.embedding_hour_dim = args.embedding_hour_dim 
        self.time_granularity = args.time_granularity
        
    def forward(self, spatial_A_trans, BS_Embedding, timestamp_seqs, base_station_seqs):
        """
        spatial_A_trans: UTGraph
        SE: pre-trained road segment representation
        src_eid_seqs: road segment sequence of RNTraj
        src_rate_seqs: moving rate of RNTraj
        """

        batchsize, max_src_len = base_station_seqs.shape
        
        id_embed = BS_Embedding[base_station_seqs]  # B, T, d

        time_embed_hour = self.time_hour_embedding(timestamp_seqs)
        time_embed_hour = self.time_hour_embedding(timestamp_seqs)

        x = torch.cat([time_embed_hour, id_embed], dim=-1)
        
        diff_noise, const_loss, x0_loss, time_loss = diff_forward_x0_constraint(net = self.diff_model, 
                                                                     x = x, 
                                                                     diffusion_hyperparams = self.diffusion_param, 
                                                                     BS_Embedding = BS_Embedding, 
                                                                     spatial_A_trans = spatial_A_trans,
                                                                     embedding_hour_dim = self.embedding_hour_dim,
                                                                     time_hour_embedding = self.time_hour_embedding.weight)
        
        return diff_noise, const_loss, x0_loss, time_loss, self.time_hour_embedding.weight

    def generate_data(self, spatial_A_trans, SE, batchsize, length, pre_dim, time_embedding_hour_matrix):
        
        """Ggenerate data"""
        
        x0 = cal_x0_from_noise_ddpm(self.diff_model, self.diffusion_param, batchsize, length, pre_dim + self.embedding_hour_dim)  # B, T, 64 + embedding_hour_dim

        # 基站处理
        x0_road = x0[:,:,self.embedding_hour_dim : ].clone()  # B, T, 64
        B, T, F = x0_road.shape

        x0_road_shape = x0_road.reshape(B*T, F)

        x0_abs = x0_road_shape.norm(dim=1)
        SE_abs = SE.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x0_road_shape, SE) / (torch.einsum('i,j->ij', x0_abs, SE_abs) + 1e-6)
        
        sim_matrix = sim_matrix.reshape(B, T, -1)  # B, T, road num    2000 * 20 = 40000 top1->top10
        sim_matrix = sim_matrix.argmax(-1)

        # 时间处理
        x0_timestamp = x0[:,:,:self.embedding_hour_dim]  # B, T, 64
        B, T, F = x0_timestamp.shape

        x0_timestamp_shape = x0_timestamp.reshape(B*T, F)
        
        x0_timestamp_abs = x0_timestamp_shape.norm(dim=1)
        time_embedding_hour_matrix_abs = time_embedding_hour_matrix.norm(dim=1)

        sim_matrix_hour = torch.einsum('ik,jk->ij', x0_timestamp_shape, time_embedding_hour_matrix) / (torch.einsum('i,j->ij', x0_timestamp_abs, time_embedding_hour_matrix_abs) + 1e-6)
        
        sim_matrix_hour = sim_matrix_hour.reshape(B, T, -1)  # B, T, road num    2000 * 20 = 40000 top1->top10
        sim_matrix_hour = sim_matrix_hour.argmax(-1)

        return sim_matrix_hour, sim_matrix
