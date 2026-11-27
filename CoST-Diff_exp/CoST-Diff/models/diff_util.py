import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from models.model_utils import ProtoHead, moln_tail_nll, sample_moln,freq_kl_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """
    return torch.normal(0, 1, size=size).cuda()

def diff_forward_x0_constraint(net, tim_decoder, traj_decoder,
                               sta_seqs, base_station_seqs, x,                      # x: [B,T,F]
                               diffusion_hyperparams, BS_Embedding,
                               spatial_A_trans, real_freq,embedding_hour_dim,
                               sc_film, cond_global=None, cond_event=None):

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["alpha_bar"]
    B, T_seq, F = x.shape

    # ===== 1. 扩散前向 =====
    diffusion_steps = torch.randint(T, size=(B,1,1), device=x.device)
    z = torch.randn_like(x)
    xt = torch.sqrt(Alpha_bar[diffusion_steps]) * x + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z

    # ===== 2. 第一次预测（无自条件）=====
    pred_noise_0 = net(xt, diffusion_steps.view(B,1), cond_global=cond_global, cond_event=cond_event)
    x0_hat_0 = (xt - torch.sqrt(1 - Alpha_bar[diffusion_steps]) * pred_noise_0) / torch.sqrt(Alpha_bar[diffusion_steps])

    # ===== 3. Self-conditioning（仅基于时间）=====
    sc_gamma, sc_beta = sc_film(x0_hat_0.detach())

    # ===== 4. 第二次预测（带自条件）=====
    pred_noise = net(xt, diffusion_steps.view(B,1), cond_global=cond_global, cond_event=cond_event, sc_gamma=sc_gamma, sc_beta=sc_beta)
    xt_noise_loss = torch.mean((pred_noise - z) ** 2)

    # ===== 5. 得到最终 x0_hat ===== 
    x0_hat = (xt - torch.sqrt(1 - Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat_time = x0_hat[:, :, :embedding_hour_dim]
    x0_hat_traj = x0_hat[:, :, embedding_hour_dim:]
    x0_loss = torch.mean((x0_hat - x) ** 2)

    # ===== 6. 空间连通性约束（soft概率）=====
    temperature = 0.3
    logits = traj_decoder(x0_hat_traj, x0_hat_time, BS_Embedding)
    prob = torch.softmax(logits / temperature, dim=-1)  # [B, T, N]

    # bad_expect: 期望非法转移概率
    bad_expect = torch.einsum('btn,nm,btm->bt', prob[:, :-1], spatial_A_trans.float(), prob[:, 1:])
    const_loss = bad_expect.mean()

    prob = torch.softmax(logits / temperature, dim=-1)  # [B, T, N]

    # 频率 KL
    freq_kl = freq_kl_loss(prob, real_freq)

    # ============================================================
    # =================== 时间建模 ===============================
    # ============================================================
    dt = torch.as_tensor(sta_seqs, dtype=x.dtype, device=x.device)  # [B, T-1]

    if not torch.isfinite(x0_hat).all():
        raise RuntimeError("🔥 x0_hat contains NaN BEFORE TimeDecoder")

    pi_pred, mu_pred, log_sig_pred = tim_decoder(x0_hat)
    pi_pred, mu_pred, log_sig_pred = pi_pred[:, :-1], mu_pred[:, :-1], log_sig_pred[:, :-1]

    nll_pred = moln_tail_nll(pi_pred, mu_pred, log_sig_pred, dt, tail_alpha=1.0)

    with torch.no_grad():
        pi_c, mu_c, log_s = tim_decoder(x)
        pi_c, mu_c, log_s = pi_c[:, :-1], mu_c[:, :-1], log_s[:, :-1]
        nll_clean = moln_tail_nll(pi_c, mu_c, log_s, dt, tail_alpha=1.0)

    sta_loss = 0.9 * nll_pred + 0.1 * nll_clean

    # return xt_noise_loss, const_loss, x0_loss, sta_loss, prob  # 返回 prob 方便生成使用
    return xt_noise_loss, const_loss, x0_loss, sta_loss, freq_kl


def cal_x0_from_noise_ddpm(net, tim_decoder, traj_decoder, diffusion_hyperparams,
                           batchsize, length, pre_dim, tim_embedding_dim, BS_Embedding, cond_global=None, cond_event=None):

    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    diff_input = std_normal((batchsize, length, pre_dim + tim_embedding_dim))

    with torch.no_grad():
        for t in range(T-1, -1, -1):
            tt = torch.full((batchsize,1), t, device=diff_input.device, dtype=torch.long)
            # predict_noise = net(diff_input, tt)
            predict_noise = net(diff_input, tt, cond_global=cond_global, cond_event=cond_event)
            coeff1 = 1 / (Alpha[t] ** 0.5)
            coeff2 = (1 - Alpha[t]) / ((1 - Alpha_bar[t]) ** 0.5)
            diff_input = coeff1 * (diff_input - coeff2 * predict_noise)

            if t > 0:
                noise = std_normal(diff_input.shape)
                sigma = ((1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t]) * (1 - Alpha[t])) ** 0.5
                diff_input += sigma * noise

    diff_input_tim  = diff_input[:, :, :tim_embedding_dim]
    diff_input_traj = diff_input[:, :, tim_embedding_dim:]

    # ===== 空间解码（soft概率生成）=====
    temperature = 0.3
    logits = traj_decoder(diff_input_traj, diff_input_tim, BS_Embedding)
    prob = torch.softmax(logits / temperature, dim=-1)  # [B, T, N]

    # 可以直接多项式采样生成轨迹，保证和训练一致
    ids = torch.stack([torch.multinomial(p, num_samples=1).squeeze(-1) for p in prob], dim=1).permute(1,0)

    # 时间解码
    pi, mu, log_sigma = tim_decoder(diff_input)
    log_sigma = log_sigma * 0.7  # 可调
    dt = sample_moln(pi, mu, log_sigma)
    dt[:, -1] = 0.0
    return dt, ids  # 返回 prob 可用于分析分布
