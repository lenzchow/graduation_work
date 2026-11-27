import math, torch, torch.nn as nn, torch.nn.functional as F
# ========== 1. 全局 / 序列级条件 ==========
class GlobalCondEncoder(nn.Module):
    """
        与轨迹整体相关、不依赖事件顺序
        areacode、district、周几、经纬度
    """
    def __init__(self, n_ac, n_dist, H, d=32):
        super().__init__()
        self.emb_ac   = nn.Embedding(n_ac, d, padding_idx=0)
        self.emb_dist = nn.Embedding(n_dist, d, padding_idx=0)
        self.emb_week = nn.Embedding(7, d)

        self.mlp_geo  = nn.Sequential(
            nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d)
        )

        self.proj = nn.Linear(d * 4, H)

    def forward(self, ac_id, dist_id, lnglat, weekday, L):
        lnglat = lnglat.float()
        
        B = ac_id.size(0)

        ac   = self.emb_ac(ac_id)
        dist = self.emb_dist(dist_id)
        geo  = self.mlp_geo(lnglat)

        if weekday.dim() == 2:
            weekday = weekday.squeeze(-1)
        week = self.emb_week(weekday)

        h = torch.cat([ac, dist, geo, week], dim=-1)  # [B,4d]
        h = self.proj(h).unsqueeze(1).expand(B, L, -1)
        return h


# ========== 2. 事件级条件 ==========
class EventCondEncoder(nn.Module):
    """
        每个事件一个条件（如 src_slot_seqs）
        早、午、晚高峰
    """
    def __init__(self, H):
        super().__init__()
        self.emb_slot = nn.Embedding(3, H)

    def forward(self, slot_ids):
        return self.emb_slot(slot_ids)   # [B,T,H]


# ========== 3. Time-of-day 条件 ==========
class TimeOfDayCond(nn.Module):
    # 起始时间、序列长度
    def __init__(self, H):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, H // 2),
            nn.SiLU(),
            nn.Linear(H // 2, H)
        )

    def forward(self, t0_min, L):
        B, dev = t0_min.size(0), t0_min.device
        steps = torch.arange(L, device=dev).unsqueeze(0)
        minutes = (t0_min.unsqueeze(1) + steps) % 1440
        x = 2 * math.pi * minutes.float() / 1440.0
        feat = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(feat)


# ========== 4. 条件总管理器 ==========
class ConditionManager(nn.Module):
    """
    只负责：
    - 组织
    - 路由
    不做融合学习
    """
    def __init__(self, n_ac, n_dist, H, d=32):
        super().__init__()
        self.global_enc = GlobalCondEncoder(n_ac, n_dist, H, d)
        self.event_enc  = EventCondEncoder(H)
        self.tod_enc    = TimeOfDayCond(H)

    def forward(self, *, ac_id, dist_id, lnglat,
                weekday, slot_ids, t0_min, L):

        cond = {}
        cond["global"] = self.global_enc(ac_id, dist_id, lnglat, weekday, L)
        cond["event"]  = self.event_enc(slot_ids)
        cond["tod"]    = self.tod_enc(t0_min, L) 
        return cond
    
class StaticCond(nn.Module):
    """
    静态 + 序列级条件编码器：
      - 起点基站的 AC / district / 坐标（静态，不随时间变化）
      - 序列级条件：weekday（0..6，整条轨迹共享）
      - 起点时间 t0_min 生成整条轨迹的 time-of-day 正弦编码
    输出: [B, L, H]
    """
    def __init__(self, n_ac, n_dist, H, d=32):
        super().__init__()
        # 行政区 & 区县/片区
        self.emb_ac   = nn.Embedding(n_ac,   d, padding_idx=0)
        self.emb_dist = nn.Embedding(n_dist, d, padding_idx=0)
        # 经纬度 → d 维
        self.mlp_geo  = nn.Sequential(
            nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d)
        )
        # 序列级条件：weekday（0..6），固定 7 类即可
        self.emb_weekday = nn.Embedding(7, d)  # 不需要 padding_idx

        # 把 [ac, dist, geo, weekday] 4 个 d 维拼一起 → H
        self.proj_static = nn.Linear(d * 4, H)

        # time-of-day（基于起始时间 + 步数）→ H
        self.mlp_tod  = nn.Sequential(
            nn.Linear(2, d), nn.SiLU(), nn.Linear(d, H)
        )

        # 初始化
        nn.init.xavier_uniform_(self.proj_static.weight, gain=0.1)
        nn.init.zeros_(self.proj_static.bias)
        nn.init.xavier_uniform_(self.mlp_tod[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp_tod[-1].bias)

    def forward(self, ac_id, dist_id, lnglat, t0_min, seq_weekday, L: int):
        lnglat = lnglat.float()
        """
        ac_id      : [B]
        dist_id    : [B]
        lnglat     : [B, 2]   (经纬度)
        t0_min     : [B]      (起点时间，单位：分钟数 0..1440)
        seq_weekday: [B] or [B,1] (0..6)
        L          : 序列长度
        """
        B, dev = ac_id.size(0), ac_id.device

        # --- 静态空间条件（起点所在区域 + 坐标） ---
        ac   = self.emb_ac(ac_id)            # [B, d]
        dist = self.emb_dist(dist_id)        # [B, d]
        geo  = self.mlp_geo(lnglat)          # [B, d]

        # --- 序列级条件：weekday ---
        if seq_weekday.dim() == 2:          # [B,1] → [B]
            seq_weekday = seq_weekday.squeeze(-1)
        weekday_emb = self.emb_weekday(seq_weekday)   # [B, d]

        # 静态部分：与时间无关的全局条件 → [B, L, H]
        static_feat = torch.cat([ac, dist, geo, weekday_emb], dim=-1)  # [B, 4d]
        h0 = self.proj_static(static_feat).unsqueeze(1).expand(B, L, -1)  # [B, L, H]

        # --- 事件级“时间-of-day 曲线”（由起点时间推整段） ---
        steps   = torch.arange(L, device=dev).unsqueeze(0)        # [1, L]
        minutes = (t0_min.unsqueeze(1) + steps) % 1440            # [B, L]
        x = 2 * math.pi * minutes.float() / 1440.0

        tod = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)   # [B, L, 2]
        h_t = self.mlp_tod(tod)                                   # [B, L, H]

        # 静态全局条件 + 时间-of-day 条件
        return h0 + h_t                                           # [B, L, H]

    
class CondFiLMStatic(nn.Module):
    def __init__(self, H, init_std=5e-4, init_gain=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(H, elementwise_affine=False)
        self.to_gb = nn.Sequential(nn.SiLU(), nn.Linear(H, 2*H))
        nn.init.normal_(self.to_gb[-1].weight, std=init_std)
        nn.init.zeros_(self.to_gb[-1].bias)
        # 全局增益（可学），起小点
        self.film_gain = nn.Parameter(torch.tensor(init_gain))

    def forward(self, cond_static):
        h = self.norm(cond_static)                          # [B,L,H]
        gamma, beta = self.to_gb(h).chunk(2, dim=-1)        # [B,L,H] views
        # 去批-时均值，避免整体漂移
        beta  = beta  - beta.mean(dim=(0,1), keepdim=True)
        gamma = gamma - gamma.mean(dim=(0,1), keepdim=True)
        # 限幅（乘性调制不要太猛）
        gamma = 0.25 * torch.tanh(gamma)                    # γ ∈ (-0.25,0.25)

        return gamma, beta, torch.sigmoid(self.film_gain)   # 将增益压到 (0,1)

# 单独一个动态 FiLM，用于时间通道自条件
class SelfCondFiLM(nn.Module):
    def __init__(self, in_dim:int, H:int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, H), nn.SiLU(), nn.Linear(H, H))
        self.to_gb = nn.Sequential(nn.SiLU(), nn.Linear(H, 2*H))
        nn.init.normal_(self.to_gb[-1].weight, std=1e-3)
        nn.init.zeros_(self.to_gb[-1].bias)

    def forward(self, x0_time_detached):        # [B,T,D_t]
        h = self.enc(x0_time_detached)          # [B,T,H]
        gamma, beta = self.to_gb(h).chunk(2, -1)
        return gamma, beta
