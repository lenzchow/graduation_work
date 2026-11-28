from collections import Counter
"""
====================================================
Trajectory & Base-Station Preprocessing Pipeline
----------------------------------------------------
功能：
1. 轨迹/基站清洗与裁剪
2. 条件变量构造（weekday / slot）
3. 序列生成（index / tim / sta / cond）
4. 图构建（transition graph）
5. 图嵌入（VGAE / Node2Vec）
----------------------------------------------------
数据流：
raw csv
  → traj_station_preprocess_single
  → extra_file (index / feat)
  → create_graph
  → gen_all
  → road_embed
====================================================
"""

"""
====================================================
Trajectory & Base-Station Preprocessing Pipeline
====================================================

功能模块：
1. 轨迹 / 基站清洗与裁剪
2. 时间与条件变量构造（weekday / slot）
3. 基站索引与特征工程
4. 序列生成（index / tim / sta / cond）
5. 转移图构建
6. 图嵌入（VGAE / Node2Vec）
7. 缓存与主流程调度

数据流：
raw csv
  → traj_station_preprocess_single
  → extra_file
  → create_graph
  → gen_all
  → road_embed

====================================================
"""

# ==================================================
# 0. Imports & Global Constants
# ==================================================

import os
import json
import copy
import pickle
import time
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.nn import VGAE

from vgae.vgae_embedding import decode, VGAE_gat
from node2vec.ge import Node2Vec


# ---------------- Geo Bounding Box ----------------
LAT_MIN, LAT_MAX = 22.705, 22.890
LON_MIN, LON_MAX = 108.2, 108.46


# ==================================================
# 1. Time & Condition Utilities
# ==================================================

def get_weekday_from_cday(c_day: int) -> int:
    """20250630 → weekday (0=Mon, 6=Sun)"""
    dt = datetime.strptime(str(int(c_day)), "%Y%m%d")
    return dt.weekday()


def build_slot_seq(tim_list, is_weekday: bool):
    """
    时间段标签：
      0 = 平峰
      1 = 早高峰 (07:00–09:00)
      2 = 晚高峰 (17:00–20:00)
    """
    if not is_weekday:
        return [0] * len(tim_list)

    slot_seq = []
    for m in tim_list:
        if 420 <= m < 540:
            slot_seq.append(1)
        elif 1020 <= m < 1200:
            slot_seq.append(2)
        else:
            slot_seq.append(0)
    return slot_seq


# ==================================================
# 2. Trajectory & Base-Station Preprocessing
# ==================================================

def filter_bs_in_box(df_bs,
                     lat_col='c_lat',
                     lon_col='c_lon'):
    """只保留落在经纬度方框内的基站"""
    mask = (
        (df_bs[lat_col] >= LAT_MIN) & (df_bs[lat_col] <= LAT_MAX) &
        (df_bs[lon_col] >= LON_MIN) & (df_bs[lon_col] <= LON_MAX)
    )
    return df_bs[mask].reset_index(drop=True)


def traj_station_preprocess_single(df_traj, df_bs, min_length, max_length):
    """
        单运营商轨迹预处理：
        1) 基站过滤
        2) 用户-天分组
        3) 长度筛选
        4) 构造 traj / tim / sta
    """
    bs_uli_set = set(df_bs['c_uli'].astype(str))
    df_traj = df_traj[df_traj['c_uli'].astype(str).isin(bs_uli_set)]

    grouped = df_traj.groupby(['c_number', 'c_day'])
    kept = [g for _, g in grouped if min_length <= len(g) <= max_length]
    control = pd.concat(kept, ignore_index=True) if kept else df_traj.iloc[0:0]

    traj_uli = set(control['c_uli'].astype(str))
    final_df_bs = df_bs[df_bs['c_uli'].astype(str).isin(traj_uli)].reset_index(drop=True)

    records = []
    for (c_number, c_day), g in control.groupby(['c_number', 'c_day']):
        g = g.sort_values('c_timestamp')
        traj = g['c_uli'].astype(str).tolist()
        traj_ts = list(zip(traj, g['c_timestamp'].astype(int)))
        records.append([c_number, c_day, traj, traj_ts])

    df = pd.DataFrame(records,
                      columns=['c_number', 'c_day', 'traj', 'traj_timestamp'])

    def ts_to_min(traj_ts):
        out = []
        for _, ts in traj_ts:
            dt = datetime.fromtimestamp(ts)
            out.append(round(dt.hour * 60 + dt.minute + dt.second / 60, 2))
        return out

    def wait_time(tim):
        return [round(tim[i+1] - tim[i], 2) for i in range(len(tim)-1)]

    df['tim'] = df['traj_timestamp'].apply(ts_to_min)
    df['sta'] = df['tim'].apply(wait_time)

    return df.reset_index(drop=True), final_df_bs


# ==================================================
# 3. Base-Station Index & Feature Engineering
# ==================================================

def build_bs_feat_with_map(df_bs, out_dir, uli_index_dict,
                           lng_col='c_lon',
                           lat_col='c_lat',
                           ac_col='c_areacode',
                           district_col='district',
                           city_col='city',
                           include_city=False):
    """
    构建并保存基站特征矩阵 bs_feat.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df_bs.copy()

    df['bs_code'] = df['c_uli'].astype(str).map(uli_index_dict)
    N = df['bs_code'].nunique()

    def vocabize(s):
        s = s.astype(str).fillna('<UNK>')
        vocab = {'<UNK>': 0}
        for v in sorted(s.unique()):
            if v != '<UNK>':
                vocab[v] = len(vocab)
        return s.map(vocab).astype('int64'), vocab

    ac_id, ac_vocab = vocabize(df[ac_col])
    dist_id, dist_vocab = vocabize(df[district_col])
    if include_city:
        city_id, city_vocab = vocabize(df[city_col])

    lng = df[lng_col].astype(float).to_numpy()
    lat = df[lat_col].astype(float).to_numpy()

    lng_mu, lng_std = lng.mean(), lng.std() or 1.0
    lat_mu, lat_std = lat.mean(), lat.std() or 1.0

    lng_z = (lng - lng_mu) / lng_std
    lat_z = (lat - lat_mu) / lat_std

    cols = ['lng_z', 'lat_z', 'ac_id', 'district_id']
    if include_city:
        cols.append('city_id')

    feat = np.zeros((N, len(cols)), dtype=np.float32)
    idx = df['bs_code'].to_numpy()

    feat[idx, 0] = lng_z
    feat[idx, 1] = lat_z
    feat[idx, 2] = ac_id
    feat[idx, 3] = dist_id
    if include_city:
        feat[idx, 4] = city_id

    np.save(os.path.join(out_dir, 'bs_feat.npy'), feat)

    json.dump(ac_vocab, open(os.path.join(out_dir, 'ac_vocab.json'), 'w'), indent=2)
    json.dump(dist_vocab, open(os.path.join(out_dir, 'district_vocab.json'), 'w'), indent=2)

    json.dump({
        'lng_mu': lng_mu,
        'lng_std': lng_std,
        'lat_mu': lat_mu,
        'lat_std': lat_std,
        'cols': cols
    }, open(os.path.join(out_dir, 'bs_feat_meta.json'), 'w'), indent=2)


def extra_file(df_bs, operator):
    """
    统一生成 index / feature 相关文件
    """
    base_dir = f'../data/{operator}/extra_file'
    os.makedirs(base_dir, exist_ok=True)

    uli_index_dict = df_bs['c_uli'].reset_index().set_index('c_uli')['index'].to_dict()
    json.dump(uli_index_dict, open(f'{base_dir}/uli_index_dict.json', 'w'), indent=2)

    index_uli_dict = {v: k for k, v in uli_index_dict.items()}
    json.dump(index_uli_dict, open(f'{base_dir}/index_uli_dict.json', 'w'), indent=2)

    index_areacode_dict = {
        str(index): str(value)
        for index, value in enumerate(df_bs['c_areacode'].tolist())
    }
    json.dump(index_areacode_dict,
              open(f'{base_dir}/index_areacode_dict.json', 'w'), indent=2)

    index_lat_lon_dict = {
        str(index): [float(row['c_lat']), float(row['c_lon'])]
        for index, (_, row) in enumerate(df_bs.iterrows())
    }
    json.dump(index_lat_lon_dict,
              open(f'{base_dir}/index_lat_lon_dict.json', 'w'), indent=2)

    build_bs_feat_with_map(df_bs, base_dir, uli_index_dict)


# ==================================================
# 4. Graph Construction
# ==================================================

def create_graph(df_traj, operator):
    os.makedirs(f'../data/{operator}/graph', exist_ok=True)

    with open(f'../data/{operator}/extra_file/uli_index_dict.json') as f:
        uli_index_dict = json.load(f)

    df_traj['traj_index'] = df_traj['traj'].apply(
        lambda t: [uli_index_dict[str(x)] for x in t]
    )

    build_graph_from_txt(df_traj, operator)


def build_graph_from_txt(traj_df, operator):
    G = nx.DiGraph()

    for _, row in traj_df.iterrows():
        t = row['traj_index']
        for i in range(len(t) - 1):
            G.add_edge(t[i], t[i+1],
                       weight=G.get_edge_data(t[i], t[i+1], {}).get('weight', 0) + 1)

    for n in G.nodes:
        G.add_edge(n, n, weight=1)

    A = nx.adjacency_matrix(G)
    np.savetxt(f'../data/{operator}/graph/graph_A.csv',
               A.todense(), delimiter=',')


# ==================================================
# 5. Sequence Generation
# ==================================================

def gen_all(df_traj, operator, min_len, max_len):
    """
    生成：
      - index_seqs.bin : {L: List[List[int]]}
      - tim_seqs.bin   : {L: List[List[float]]}
      - sta_seqs.bin   : {L: List[List[float]]}
      - length_distri.npy
      - slot_seqs.bin  : {L: List[List[int]]}       # 事件条件：时间段 0/1/2
      - seq_cond.bin   : {L: List[int]}             # 序列条件：weekday 0..6
    """
    # 确保 gen_all 目录存在
    os.makedirs(f'../data/{operator}/gen_all', exist_ok=True)

    # 读取 uli -> index 映射表
    with open(f'../data/{operator}/extra_file/uli_index_dict.json', 'r', encoding='utf-8') as file:
        uli_index_dict = json.load(file)

    result_dict = {}   # 轨迹 index 序列
    tim_dict    = {}   # tim 序列
    sta_dict    = {}   # sta 序列

    slot_dict   = {}   # 事件级条件：时间段标签序列
    seq_cond_dict = {} # 序列级条件：weekday 标签（0..6）

    # 轨迹最大长度
    max_length = df_traj['traj'].apply(len).max()

    result_list = [0] * (max_len + 1)
    tim_list    = [0] * (max_len + 1)
    sta_list    = [0] * (max_len + 1)

    # 遍历每条轨迹（每一行 = 一条 用户-天 轨迹）
    for _, row in df_traj.iterrows():
        traj = row['traj']              # List[uli str]
        tim  = row.get('tim', [])       # List[分钟数 float]
        sta  = row.get('sta', [])       # List[等待时间 float]
        c_day = row['c_day']            # 形如 20250630 的 int

        trajlen = len(traj)
        if trajlen == 0:
            continue

        # 长度过滤，用和原始代码一样的 min_len/max_len 约束
        if not (min_len <= trajlen <= max_len):
            continue

        # uli → 连续 index
        mapped_traj = [uli_index_dict[str(uli)] for uli in traj]

        # === 序列级条件：weekday (0..6) ===
        weekday = get_weekday_from_cday(c_day)
        # 周一-周五是工作日
        is_weekday = (weekday <= 4)

        # === 事件级条件：时间段标签序列 slot_seq ===
        # tim 的长度应该和 trajlen 相同
        if isinstance(tim, list) and len(tim) == trajlen:
            slot_seq = build_slot_seq(tim, is_weekday=is_weekday)
        else:
            # tim 缺失或长度对不上时，退化为全 0 平峰
            slot_seq = [0] * trajlen

        # --- 按长度 L 归类 ---
        result_dict.setdefault(trajlen, []).append(mapped_traj)
        tim_dict.setdefault(trajlen, []).append(tim)
        sta_dict.setdefault(trajlen, []).append(sta)

        slot_dict.setdefault(trajlen, []).append(slot_seq)
        seq_cond_dict.setdefault(trajlen, []).append(int(weekday))  # 每条轨迹一个 int label

    # 长度分布数组（和原来一样）
    for key in result_dict:
        if min_len <= key <= max_len:
            result_list[key] = len(result_dict[key])
            tim_list[key]    = len(tim_dict.get(key, []))
            sta_list[key]    = len(sta_dict.get(key, []))

    result_array = np.array(result_list)
    tim_array    = np.array(tim_list)
    sta_array    = np.array(sta_list)

    # 保存文件（原有的三个）
    with open(f'../data/{operator}/gen_all/index_seqs.bin', 'wb') as file:
        pickle.dump(result_dict, file)
    np.save(f'../data/{operator}/gen_all/length_distri.npy', result_array)

    with open(f'../data/{operator}/gen_all/tim_seqs.bin', 'wb') as file:
        pickle.dump(tim_dict, file)

    with open(f'../data/{operator}/gen_all/sta_seqs.bin', 'wb') as file:
        pickle.dump(sta_dict, file)

    # 新增：事件级条件（slot_seqs.bin）和序列级条件（seq_cond.bin）
    with open(f'../data/{operator}/gen_all/slot_seqs.bin', 'wb') as file:
        pickle.dump(slot_dict, file)

    with open(f'../data/{operator}/gen_all/seq_cond.bin', 'wb') as file:
        pickle.dump(seq_cond_dict, file)

    return result_dict, tim_dict, sta_dict, slot_dict, seq_cond_dict


# ==================================================
# 6. Graph Embedding
# ==================================================

# def road_embed(df_traj, df_bs, embed_size, operator):
#     with open(f'../data/{operator}/extra_file/uli_index_dict.json') as f:
#         uli_index_dict = json.load(f)

#     df_traj['traj_index'] = df_traj['traj'].apply(
#         lambda t: [uli_index_dict[str(x)] for x in t]
#     )

#     transitions = []
#     for _, r in df_traj.iterrows():
#         t = r['traj_index']
#         transitions += [(t[i], t[i+1]) for i in range(len(t)-1)]

#     edges = torch.tensor(transitions).T
#     X = lac_feature(df_bs)

#     data = Data(x=torch.tensor(X, dtype=torch.float),
#                 edge_index=edges)

#     model = VGAE(VGAE_gat(data.num_node_features, embed_size, [2, 2, 1]))
#     z = decode(model, 100, embed_size, data, torch.device('cpu'))

#     with open(f'../data/{operator}/graph/road_embed_vgae.txt', 'w') as f:
#         f.write(f"{z.shape[0]} {z.shape[1]}\n")
#         for i, row in enumerate(z):
#             f.write(f"{i} " + " ".join(map(str, row.tolist())) + "\n")

def road_embed(df_traj, df_bs, embed_size, operator):
    # 读取uli_index_dict.json数据
    with open(f'../data/{operator}/extra_file/uli_index_dict.json', 'r', encoding='utf-8') as file:
        uli_index_dict = json.load(file)
        
    lac_features = lac_feature(df_bs)
    
    transition_counts = Counter()
    for _, row in df_traj.iterrows():
        traj_seq = row['traj_index']
        for i in range(len(traj_seq) - 1):
            transition_counts[(traj_seq[i], traj_seq[i + 1])] += 1

    unique_transitions_list = list(transition_counts)
    unique_transitions_list_weighted = [
        (transition, count) for transition, count in transition_counts.items()
    ]
    
    vage_embed(lac_features, unique_transitions_list, embed_size, operator)
    node2vec(unique_transitions_list_weighted, embed_size, operator)

def vage_embed(lac_features, unique_transitions_list, embed_size, operator):
    edges1 = [list(map(int,[line[0],line[1]])) for line in unique_transitions_list]
    # edges2 = [list(map(int,[line.split()[1],line.split()[0]])) for line in unique_transitions_list]
    # edges = edges1+edges2
    edges = edges1

    edges = torch.tensor(edges,dtype=torch.long)
    edges = edges.transpose(0,1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_size = embed_size
    data = Data(x= torch.tensor(lac_features,dtype=torch.float),edge_index=edges) # 修改其中的features

    num_features = data.num_node_features

    model = VGAE(VGAE_gat(num_features, embed_size,[2,2,1]))  # new line 修改其中的num_features
    model = model.to(device)
    z_gat= decode(model,100,embed_size,data,device) # 修改其中的data
    
    # 打开一个文本文件以写入模式
    with open(f'../data/{operator}/graph/road_embed_vgae.txt', 'w') as file:
        first_line = f"{z_gat.shape[0]} {z_gat.shape[1]}\n"
        file.write(first_line)

        # 遍历每一行，对每行元素求和后写入
        for index, row in enumerate(z_gat):
            row_value = " ".join(str(value.item()) for value in row)
            line = f"{index} {row_value}\n"
            file.write(line)
    
    
def node2vec(unique_transitions_list_weighted, out_dims, operator):
    G = nx.DiGraph()
    for transition in unique_transitions_list_weighted:
        node, weight = transition
        G.add_edge(node[0], node[1], weight=weight)
    
    embed_size = out_dims
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=2, workers=1, use_rejection_sampling=0)
    model.train(embed_size=embed_size,window_size = 5, epochs = 30)
    embeddings=model.get_embeddings()
    
    # 打开一个文本文件以写入模式
    with open(f'../data/{operator}/graph/road_embed_node2vec.txt', 'w') as file:
        first_line = f"{len(embeddings.keys())} {embed_size}\n"
        file.write(first_line)

        # 遍历字典的键值对
        for key, row in embeddings.items():
            row_value = " ".join(str(value.item()) for value in row)
            line = f"{key} {row_value}\n"
            file.write(line)
def lac_feature(df_bs):
    onehot_lac = pd.get_dummies(df_bs['c_lac'].astype(str)).to_numpy()
    lat = np.deg2rad(df_bs['c_lat'].to_numpy())
    lon = np.deg2rad(df_bs['c_lon'].to_numpy())
    geo = np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)], 1)
    ac = pd.get_dummies(df_bs['c_areacode'].astype(str)).to_numpy()
    return np.hstack([onehot_lac, geo, ac]).astype(np.float32)


# ==================================================
# 7. Cache & Main
# ==================================================

def get_processed_or_build_single(op_id, df_traj, df_bs,
                                  min_length, max_length,
                                  root='./handled_data'):
    cache_dir = os.path.join(root, f'op{op_id}')
    os.makedirs(cache_dir, exist_ok=True)

    traj_p = os.path.join(cache_dir, 'traj.pkl')
    bs_p = os.path.join(cache_dir, 'bs.pkl')

    if os.path.exists(traj_p) and os.path.exists(bs_p):
        return pd.read_pickle(traj_p), pd.read_pickle(bs_p)

    df_t, df_b = traj_station_preprocess_single(
        df_traj, df_bs, min_length, max_length
    )
    df_t.to_pickle(traj_p)
    df_b.to_pickle(bs_p)
    return df_t, df_b


def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    repo_root = Path(__file__).resolve().parents[3]
    split_root = repo_root / "privacy_eval" / "local" / "splits" / "CoST-Diff_exp" / "CoST-Diff"

    embed_size = 128
    min_length, max_length = 10, 60
    prepared = 0

    for operator in ("0", "1", "3"):
        split_dir = split_root / f"operator{operator}"
        train_path = split_dir / "train.csv"
        holdout_path = split_dir / "holdout.csv"
        station_path = split_dir / "base_station_train.csv"
        if not train_path.is_file() or not holdout_path.is_file() or not station_path.is_file():
            raise FileNotFoundError(
                f"Missing split files for operator{operator}; run privacy_eval/prepare_all_splits.sh first"
            )

        df_traj = pd.read_csv(train_path, dtype={"c_uli": str})
        df_bs = pd.read_csv(station_path, dtype={"c_uli": str})
        df_bs = filter_bs_in_box(df_bs)

        df_t, df_b = traj_station_preprocess_single(
            df_traj, df_bs, min_length, max_length
        )
        tag = f"operator{operator}"
        data_root = Path("../data") / tag
        if data_root.exists():
            shutil.rmtree(data_root)

        if df_t.empty or df_b.empty:
            print(f"Skipping operator{operator}: no eligible training trajectories or stations")
            continue

        (data_root / "extra_file").mkdir(parents=True, exist_ok=True)
        (data_root / "graph").mkdir(parents=True, exist_ok=True)

        extra_file(df_b, tag)
        create_graph(df_t, tag)
        gen_all(df_t, tag, min_length, max_length)
        gen_dir = data_root / "gen_all"
        np.save(gen_dir / "length_distri_train.npy",
                np.load(gen_dir / "length_distri.npy"))

        holdout = pd.read_csv(holdout_path, dtype={"c_uli": str})
        station_vocab = set(df_b["c_uli"].astype(str))
        eligible_holdout_lengths = []
        total_holdout_tracks = 0
        for _, group in holdout.groupby(["c_number", "c_day"], sort=False):
            total_holdout_tracks += 1
            length = len(group)
            locations = set(group["c_uli"].astype(str))
            if min_length <= length <= max_length and locations.issubset(station_vocab):
                eligible_holdout_lengths.append(length)

        holdout_distribution = np.zeros(max_length + 1, dtype=np.int64)
        for length, count in Counter(eligible_holdout_lengths).items():
            holdout_distribution[length] = count
        if holdout_distribution.sum() == 0:
            raise RuntimeError(
                f"No eligible holdout trajectories remain for operator{operator} "
                "within the configured length range and training station vocabulary"
            )
        np.save(gen_dir / "length_distri_holdout.npy", holdout_distribution)
        np.save(gen_dir / "length_distri.npy", holdout_distribution)

        print(
            f"{tag}: train trajectories={len(df_t):,}, "
            f"eligible holdout={len(eligible_holdout_lengths):,}/"
            f"{total_holdout_tracks:,}, stations={len(df_b):,}"
        )
        road_embed(df_t, df_b, embed_size, tag)
        prepared += 1

    if prepared == 0:
        raise RuntimeError("No operator artifacts were produced from the training splits")


if __name__ == "__main__":
    main()
