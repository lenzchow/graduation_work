import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats

# irank
def personal_topk_freq(trajs, user_col='c_number', loc_col='c_uli', K=100):
    """
    trajs: DataFrame，每行一条记录（至少包含 user 和 loc），或
           可迭代的(用户id -> 该用户轨迹序列list)映射，按需改下取法。
    返回：所有用户的Top-K相对频率拼接成的一维数组（长度=用户数*K）
    """
    all_freqs = []
    # ——若是 DataFrame——
    for uid, g in trajs.groupby(user_col):
        locs = g[loc_col].tolist()
        cnt = Counter(locs)
        freq = np.array(sorted(cnt.values(), reverse=True)[:K], dtype=float)
        if freq.sum() > 0:
            freq = freq / freq.sum()       # 个人内归一化
        if len(freq) < K:
            freq = np.pad(freq, (0, K-len(freq)))
        all_freqs.append(freq)
    return np.concatenate(all_freqs)

# 计算 I-rank 分布（0~1 区间直方图，转概率并加平滑）
def hist_prob(x, bins=100):
    h, _ = np.histogram(x, bins=bins, range=(0,1))
    p = h.astype(float); p /= p.sum() + 1e-12
    p = (p + 1e-12); p /= p.sum()  # 平滑
    return p

#rsc
def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(path, header=None)  # 不使用表头（假设文件没有表头）
    print('df.shape',df.shape)
    # 将 pandas DataFrame 转换为 numpy 数组
    A = df.to_numpy()
    # 处理矩阵：将非零元素替换为 1，零元素替换为 1e-10
    A[A != 0.] = 1.
    A[A == 0.] = 1e-10

    # 返回处理后的邻接矩阵
    A1 = A
    # print(A1.shape)

    return A1


# 计算单条轨迹的RSC指标
def calculate_single_rsc(trajectory, graph):
    T = len(trajectory)
    numerator = 0
    count = 0
    for t in range(T - 1):
        point_t = trajectory[t]
        point_t_plus_1 = trajectory[t + 1]
        # if point_t >= 22685 or point_t_plus_1 >= 22685:
        #     continue
        if (graph[point_t-1][point_t_plus_1-1] != 1e-10):
            numerator += 1
        count += 1
    rsc = (numerator / (T - 1)) * 100
    return rsc

# f_jsd

# def get_topk_visits_f(trajs, k):
#     topk_visits_loc = []
#     topk_visits_freq = []
#     for traj in trajs:
#         topk = Counter(traj).most_common(k)
#         for i in range(len(topk), k):
#             # supplement with (loc=-1, freq=0)
#             topk += [(-1, 0)]
#         loc = [l for l, _ in topk]
#         freq = [f for _, f in topk]
#         loc = np.array(loc, dtype=int)
#         freq = np.array(freq, dtype=float) / trajs.shape[1]
#         topk_visits_loc.append(loc)
#         topk_visits_freq.append(freq)
#     topk_visits_loc = np.array(topk_visits_loc, dtype=int)
#     topk_visits_freq = np.array(topk_visits_freq, dtype=float)
#     return topk_visits_loc, topk_visits_freq

# def get_overall_topk_visits_freq(trajs, k):
#         _, topk_visits_freq = get_topk_visits_f(trajs, k)
#         mn = np.mean(topk_visits_freq, axis=0)
#         return mn / np.sum(mn)

def get_topk_visits_f(trajs, k):
    topk_visits_loc = []
    topk_visits_freq = []
    for traj in trajs:
        topk = Counter(traj).most_common(k)
        for i in range(len(topk), k):
            # supplement with (loc=-1, freq=0)
            topk += [(-1, 0)]
        loc = [l for l, _ in topk]
        freq = [f for _, f in topk]
        loc = np.array(loc, dtype=int)
        freq = np.array(freq, dtype=float) / len(traj)  # 使用当前轨迹的长度
        topk_visits_loc.append(loc)
        topk_visits_freq.append(freq)
    return topk_visits_loc, topk_visits_freq


def get_overall_topk_visits_freq(trajs, k):
    topk_visits_loc_list, topk_visits_freq_list = get_topk_visits_f(trajs, k)
    # 将每个轨迹的频率列表转换为numpy数组
    topk_visits_freq_arr = np.array(topk_visits_freq_list)
    mn = np.mean(topk_visits_freq_arr, axis=0)
    return mn / np.sum(mn)

#l_jsd
def get_visits(trajs,max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(max_locs), dtype=float)
        for traj in trajs:
            for t in traj:
                visits[t] += 1
        visits = visits / np.sum(visits)
        return visits
    
def get_topk_visits_l(visits, K):
        """
        get top-k visits and the corresponding locations
        :param trajs:
        :param K:
        :return:
        """
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [locs_visits[i][0] for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs
    
#p_jsd
# def get_periodicity(trajs, length):
#         """
#         stat how many repetitions within a single trajectory
#         :param trajs:
#         :return:
#         """
#         reps = []
#         for traj in trajs:
#             reps.append(float(len(set(traj)))/ length )
#         reps = np.array(reps, dtype=float)
#         return reps
def get_periodicity(trajs, lengths):
    """
    stat how many repetitions within a single trajectory
    :param trajs: 轨迹列表，每个元素是一个轨迹（列表）
    :param lengths: 每个轨迹对应的长度列表
    :return:
    """
    reps = []
    for traj, length in zip(trajs, lengths):
        reps.append(float(len(set(traj))) / length)
    reps = np.array(reps, dtype=float)
    return reps

#in common
def arr_to_distribution(arr, min, max, bins):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    # distribution, base = np.histogram(
    #     arr, np.arange(
    #         min, max, float(
    #             max - min) / bins))
    bin_edges = np.linspace(min, max, bins + 1)  # 生成 bins+1 个边界
    distribution, base = np.histogram(arr, bins=bin_edges)  # 确保 bins 个区间
    return distribution, base[:-1]

def get_js_divergence(p1, p2):
    """
    calculate the Jensen-Shanon Divergence of two probability distributions
    :param p1:
    :param p2:
    :return:
    """
    # normalize
    p1 = p1 / (p1.sum()+1e-14)
    p2 = p2 / (p2.sum()+1e-14)
    m = (p1 + p2) / 2
    js = 0.05 * scipy.stats.entropy(p1, m) + \
        0.05 * scipy.stats.entropy(p2, m)
    return js 

# arealoc
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ---------- 1) dict -> GPS 数组（列顺序= [lat, lon]） ----------
def build_GPS_from_dict(index_lat_lon_dict):
    items = sorted(((int(k), v[0], v[1]) for k, v in index_lat_lon_dict.items()), key=lambda x: x[0])
    max_idx = items[-1][0]
    GPS = np.full((max_idx + 1, 2), np.nan, dtype=float)  # [lat, lon]
    for i, lat, lon in items:
        GPS[i, 0] = float(lat)
        GPS[i, 1] = float(lon)
    return GPS  # shape (N,2), 列= [lat, lon]

# ---------- 2) 基础距离 ----------
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = phi2 - phi1
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def traj_travel_distance_idx(traj_idx, GPS_latlon):
    if len(traj_idx) < 2: return 0.0
    dist = 0.0
    for a, b in zip(traj_idx[:-1], traj_idx[1:]):
        a = int(a); b = int(b)
        if a < 0 or b < 0 or a >= len(GPS_latlon) or b >= len(GPS_latlon): continue
        lat1, lon1 = GPS_latlon[a, 0], GPS_latlon[a, 1]
        lat2, lon2 = GPS_latlon[b, 0], GPS_latlon[b, 1]
        if np.isnan(lat1) or np.isnan(lat2): continue
        dist += haversine_m(lon1, lat1, lon2, lat2)
    return float(dist)

def traj_radius_gyration_idx(traj_idx, GPS_latlon):
    pts = []
    for i in traj_idx:
        i = int(i)
        if 0 <= i < len(GPS_latlon) and not np.isnan(GPS_latlon[i,0]):
            pts.append(GPS_latlon[i])
    if not pts: return 0.0
    pts = np.asarray(pts)  # [lat, lon]
    lat0, lon0 = pts[:,0].mean(), pts[:,1].mean()
    # 度->米的局部线性近似
    X = np.column_stack([
        (pts[:,1] - lon0) * (111320 * np.cos(np.deg2rad(lat0))),  # lon->m
        (pts[:,0] - lat0) * 110540                                 # lat->m
    ])
    c = X.mean(axis=0)
    rg = np.sqrt(((X - c)**2).sum(axis=1).mean())
    return float(rg)

# ---------- 3) 分布+JSD ----------
def distance_and_radius_jsd_indices(real_trajs, gen_trajs, index_lat_lon_dict,
                                    arr_to_distribution, get_js_divergence,
                                    bins=100):
    GPS_latlon = build_GPS_from_dict(index_lat_lon_dict)  # 列 = [lat, lon]

    real_dist = np.array([traj_travel_distance_idx(t, GPS_latlon) for t in real_trajs], float)
    gen_dist  = np.array([traj_travel_distance_idx(t, GPS_latlon) for t in gen_trajs ], float)
    real_rg   = np.array([traj_radius_gyration_idx(t, GPS_latlon) for t in real_trajs], float)
    gen_rg    = np.array([traj_radius_gyration_idx(t, GPS_latlon) for t in gen_trajs ], float)

    # 距离 JSD
    lo, hi = float(min(real_dist.min(), gen_dist.min())), float(max(real_dist.max(), gen_dist.max()))
    if lo == hi: hi = lo + 1.0
    p_dist, _ = arr_to_distribution(real_dist, lo, hi, bins)
    q_dist, _ = arr_to_distribution(gen_dist,  lo, hi, bins)
    jsd_dist = get_js_divergence(p_dist, q_dist)

    # 半径 JSD
    lo, hi = float(min(real_rg.min(), gen_rg.min())), float(max(real_rg.max(), gen_rg.max()))
    if lo == hi: hi = lo + 1.0
    p_rg, _ = arr_to_distribution(real_rg, lo, hi, bins)
    q_rg, _ = arr_to_distribution(gen_rg,  lo, hi, bins)
    jsd_rg = get_js_divergence(p_rg, q_rg)

    return {"Distance": jsd_dist, "Radius": jsd_rg}, (real_dist, gen_dist, real_rg, gen_rg)


# 基站jsd
import numpy as np
from scipy.stats import entropy

def calculate_jsd(p, q):
    """
    计算两个概率分布之间的Jensen - Shannon散度
    :param p: 第一个概率分布
    :param q: 第二个概率分布
    :return: JSD值
    """
    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return jsd/10

def get_base_station_distribution(trajectories):
    """
    统计所有轨迹数据中每个基站的分布概率
    :param trajectories: 轨迹数据列表，每个元素是一条轨迹
    :return: 基站分布概率数组
    """
    all_points = [point for trajectory in trajectories for point in trajectory]
    unique, counts = np.unique(all_points, return_counts=True)
    probabilities = counts / len(all_points)
    return unique, probabilities

def align_distributions(unique1, probs1, unique2, probs2):
    """
    对齐两个概率分布，确保它们的长度相同
    :param unique1: 第一个分布中出现的唯一基站ID数组
    :param probs1: 第一个分布的基站分布概率数组
    :param unique2: 第二个分布中出现的唯一基站ID数组
    :param probs2: 第二个分布的基站分布概率数组
    :return: 对齐后的概率分布
    """
    all_unique = np.unique(np.concatenate([unique1, unique2]))
    p_aligned = np.zeros(len(all_unique))
    q_aligned = np.zeros(len(all_unique))
    for i, base_station in enumerate(all_unique):
        if base_station in unique1:
            index = np.where(unique1 == base_station)[0][0]
            p_aligned[i] = probs1[index]
        if base_station in unique2:
            index = np.where(unique2 == base_station)[0][0]
            q_aligned[i] = probs2[index]
    return p_aligned, q_aligned

##########################
# 时间部分计算
##########################
# --------- JSD计算 ----------
def JSD(P_A, P_B):
    epsilon = 1e-8
    P_A = P_A + epsilon
    P_B = P_B + epsilon
    P_merged = 0.5 * (P_A + P_B)
    
    kl_PA_PM = np.sum(P_A * np.log(P_A / P_merged))
    kl_PB_PM = np.sum(P_B * np.log(P_B / P_merged))
    
    return 0.5 * (kl_PA_PM + kl_PB_PM)

# --------- 直方图转概率 ----------
def probability(count, interval=None, bins=50):
    p, x = np.histogram(count, bins=bins, range=interval)
    x = (x[:-1] + x[1:]) / 2
    p = p / np.sum(p) if np.sum(p) > 0 else np.ones_like(p) / len(p)
    return x, p

# --------- Duration ----------
def calc_duration_jsd(real_sta, gen_sta):
    # real_sta / gen_sta: list[list]
    real_flat = np.concatenate(real_sta)
    gen_flat = np.concatenate(gen_sta)

    # 过滤负值或0
    real_flat = real_flat[real_flat > 0]
    gen_flat = gen_flat[gen_flat > 0]

    unit = (np.percentile(real_flat, 95) - np.min(real_flat)) / 200
    interval = (np.min(real_flat) - unit, np.percentile(real_flat, 95) + unit)

    _, p_real = probability(real_flat, interval)
    _, p_gen = probability(gen_flat, interval)

    return JSD(p_gen, p_real)

# --------- Move ----------
def calc_move_jsd(real_tim, gen_tim, slot=60):
    def counts(tim_list):
        k = 1440 // slot
        c = np.zeros(k, dtype=float)
        for traj in tim_list:
            s = ((np.asarray(traj) % 1440) // slot).astype(int)
            for v in s: c[v] += 1
        return c
    pr = counts(real_tim); pr = pr / pr.sum()
    pg = counts(gen_tim);  pg = pg / pg.sum()
    return JSD(pg + 1e-12, pr + 1e-12)

# --------- Stay ----------
import numpy as np
from collections import defaultdict
import random

def calc_stay_mse(real_tim, real_sta, gen_tim, gen_sta, slot=60):
    def slot_avg_wait(tim_list, sta_list):
        k = 1440 // slot
        sums = np.zeros(k); cnts = np.zeros(k)
        for t, w in zip(tim_list, sta_list):
            t = np.asarray(t); w = np.asarray(w)
            if len(t) <= 1 or len(w) == 0: 
                continue
            s = ((t[:-1] % 1440) // slot).astype(int)
            m = min(len(s), len(w))              # 严格对齐
            for si, wi in zip(s[:m], w[:m]):
                sums[si] += wi; cnts[si] += 1
        avg = np.zeros_like(sums, dtype=float)
        nz = cnts > 0
        avg[nz] = sums[nz] / cnts[nz]
        return avg, cnts

    avg_r, cnt_r = slot_avg_wait(real_tim, real_sta)
    avg_g, cnt_g = slot_avg_wait(gen_tim,  gen_sta)

    mask = (cnt_r > 0) & (cnt_g > 0)        # 两侧均有样本
    if not mask.any(): 
        return float('nan')
    return float(np.mean((avg_r[mask] - avg_g[mask])**2))


def _traj_len(tim, sta):
    # 以等待序列长度为准；若 tim 比 sta 多 1（常见），两者等价
    return len(sta)

def select_real_subset_by_gen_lengths(real_tim, real_sta, gen_tim, gen_sta, seed=0, replace=True):
    """
    根据“生成轨迹的长度分布”从真实轨迹中采样相同长度&数量的子集。
    replace=True 表示当真实该长度不足时放回采样，保证数量严格匹配。
    """
    rng = random.Random(seed)

    # 1) 按长度建桶
    real_buckets = defaultdict(list)
    for i, (t, s) in enumerate(zip(real_tim, real_sta)):
        real_buckets[_traj_len(t, s)].append(i)

    gen_need = defaultdict(int)
    gen_buckets = defaultdict(list)
    for j, (t, s) in enumerate(zip(gen_tim, gen_sta)):
        L = _traj_len(t, s)
        gen_need[L] += 1
        gen_buckets[L].append(j)

    # 2) 逐长度采样
    chosen_idx = []
    info = {}
    for L, need in gen_need.items():
        pool = real_buckets.get(L, [])
        have = len(pool)
        info[L] = {"need": need, "have": have, "picked": 0, "replaced": False}

        if have == 0:
            # 真实集中完全没有该长度，跳过（或可改成就近长度策略）
            continue

        if replace:
            picks = [rng.choice(pool) for _ in range(need)]
            info[L]["picked"] = len(picks)
            info[L]["replaced"] = (need > have)
        else:
            k = min(need, have)
            picks = rng.sample(pool, k)
            info[L]["picked"] = k

        chosen_idx.extend(picks)

    # 3) 返回真实子集
    r_tim_sel = [real_tim[i] for i in chosen_idx]
    r_sta_sel = [real_sta[i] for i in chosen_idx]
    return r_tim_sel, r_sta_sel, info

def calc_stay_mse_matched(real_tim, real_sta, gen_tim, gen_sta, slot=60, seed=0, replace=True):
    r_tim_sel, r_sta_sel, info = select_real_subset_by_gen_lengths(
        real_tim, real_sta, gen_tim, gen_sta, seed=seed, replace=replace
    )
    mse = calc_stay_mse(r_tim_sel, r_sta_sel, gen_tim, gen_sta, slot=60)
    return mse, info