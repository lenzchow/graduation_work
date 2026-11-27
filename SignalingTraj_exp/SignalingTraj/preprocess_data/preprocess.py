from collections import Counter
import pandas as pd
import copy
import networkx as nx
import os
import json
import pickle
from vgae.vgae_embedding import decode, VGAE_gat
import torch
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import VGAE
from torch_geometric.data import Data
import numpy as np
import numpy as np
from node2vec.ge.classify import read_node_label, Classifier
from node2vec.ge import Node2Vec
import time
import shutil
from pathlib import Path
from datetime import datetime


def traj_station_preprocess(df_traj, df_bs, min_length, max_length):
    
    # 统计c_uli列中每个值出现的次数
    count_result = df_traj['c_uli'].value_counts()

    # 筛选出出现次数 ≥ 30的值
    filtered_result = count_result[count_result >= 30]
    # 以上代码无用，每个基站至少被200个用户经过了，所以不存在出现次数 <= 30的基站
    
    # 在df_traj中筛选出c_uli列的值在filtered_values中的行
    df_traj_fliter_station = copy.copy(df_traj)

    # 使用filtered_result(过筛的基站)的索引 对轨 迹进行筛选
    df_traj_fliter_station= df_traj_fliter_station[df_traj_fliter_station['c_uli'].isin(filtered_result.index)]

    # 将基站表df_bs中的c_uli列转换为列表
    bs_uli_list = df_bs['c_uli'].tolist()

    # 在df_traj_fliter_station中筛选，只保留c_uli在bs_uli_list中的行
    df_traj_fliter_station = df_traj_fliter_station[df_traj_fliter_station['c_uli'].isin(bs_uli_list)]

    # 假设traj_fliter_station_df是你当前的DataFrame
    grouped = df_traj_fliter_station.groupby(['c_number', 'c_day'])

    filtered_groups = []
    for group_key, group_data in grouped:
        group_length = len(group_data)
        if min_length <= group_length <= max_length:
            filtered_groups.append(group_data)

    control_traj_df = pd.concat(filtered_groups, ignore_index=True)

    # 筛选轨迹中所有出现过的c_uli
    traj_uli_list = control_traj_df['c_uli'].unique().tolist()
    # 最后对基站文件进行过筛
    final_df_bs = df_bs[df_bs['c_uli'].isin(traj_uli_list)]
    
    # 假设traj_fliter_station_df是你当前的DataFrame
    grouped_fliter = control_traj_df.groupby(['c_number', 'c_day'])

    new_data = []
    for (c_number, c_day), group in grouped_fliter:
        sorted_group = group.sort_values('c_timestamp')
        traj = list(sorted_group['c_uli'])
        traj_timestamp = list(zip(sorted_group['c_uli'], sorted_group['c_timestamp']))
        new_data.append([c_number, c_day, traj, traj_timestamp])

    final_traj_df = pd.DataFrame(new_data, columns=['c_number', 'c_day', 'traj', 'traj_timestamp'])


    # 提取traj中list的第一项
    first_items = final_traj_df['traj'].apply(lambda x: x[0] if x else None)

    df_0 = final_traj_df[first_items.str.startswith('0')].copy()
    df_1 = final_traj_df[first_items.str.startswith('1')].copy()
    df_3 = final_traj_df[first_items.str.startswith('3')].copy()

    if not df_0.empty:
        df_0.loc[:, 'tim'] = df_0.apply(timestamp_to_minutes, axis=1)
        df_0.loc[:, 'sta'] = df_0.apply(compute_waiting_time, axis=1)

    if not df_1.empty:
        df_1.loc[:, 'tim'] = df_1.apply(timestamp_to_minutes, axis=1)
        df_1.loc[:, 'sta'] = df_1.apply(compute_waiting_time, axis=1)
    
    if not df_3.empty:
        df_3.loc[:, 'tim'] = df_3.apply(timestamp_to_minutes, axis=1)
        df_3.loc[:, 'sta'] = df_3.apply(compute_waiting_time, axis=1)
    

    # 按照0、1、3开头分类
    df_0_bs = final_df_bs[final_df_bs['c_uli'].str.startswith('0')]
    df_0_bs = df_0_bs.reset_index(drop=True)  # 重置df_0_bs的索引

    df_1_bs = final_df_bs[final_df_bs['c_uli'].str.startswith('1')]
    df_1_bs = df_1_bs.reset_index(drop=True)  # 重置df_1_bs的索引

    df_3_bs = final_df_bs[final_df_bs['c_uli'].str.startswith('3')]
    df_3_bs = df_3_bs.reset_index(drop=True)  # 重置df_3_bs的索引
    
    return df_0, df_1, df_3, df_0_bs, df_1_bs, df_3_bs


def timestamp_to_minutes(row):
    new_ts = []
    for _, ts in row['traj_timestamp']:  # 每个元素是 (cell_id, timestamp)
        dt = datetime.fromtimestamp(ts)
        minutes = dt.hour * 60 + dt.minute + dt.second / 60
        new_ts.append(round(minutes, 2))
    return new_ts

def compute_waiting_time(row):
    tim_list = row['tim']
    if not isinstance(tim_list, list) or len(tim_list) < 2:
        return []
    return [round(tim_list[i+1] - tim_list[i], 2) for i in range(len(tim_list) - 1)]

# extra_file_handle
def extra_file(df_bs, operator):
    uli_index_dict = df_bs['c_uli'].reset_index().set_index('c_uli')['index'].to_dict()
    with open(f'../data/{operator}/extra_file/uli_index_dict.json', 'w', encoding='utf-8') as file:
        json.dump(uli_index_dict, file, ensure_ascii=False, indent=4)

    # 反转字典，使得键为原来的索引，值为原来的c_uli
    index_uli_dict = {k: v for k, v in df_bs['c_uli'].to_dict().items()}
    with open(f'../data/{operator}/extra_file/index_uli_dict.json', 'w', encoding='utf-8') as file:
        json.dump(index_uli_dict, file, ensure_ascii=False, indent=4)

    index_areacode_dict = df_bs['c_areacode'].to_dict()
    with open(f'../data/{operator}/extra_file/index_areacode_dict.json', 'w', encoding='utf-8') as file:
        json.dump(index_areacode_dict, file, ensure_ascii=False, indent=4)

    index_lat_lon_dict = {}
    for index, row in df_bs.iterrows():
        lat = row['c_lat']
        lon = row['c_lon']
        index_lat_lon_dict[index] = (lat, lon)
    with open(f'../data/{operator}/extra_file/index_lat_lon_dict.json', 'w', encoding='utf-8') as f:
        json.dump(index_lat_lon_dict, f, ensure_ascii=False, indent=4)
    
    
    
# create graph
def create_graph(df_traj, operator):
    # 读取uli_index_dict.json数据
    with open(f'../data/{operator}/extra_file/uli_index_dict.json', 'r', encoding='utf-8') as file:
        uli_index_dict = json.load(file)
    
    # 应用映射函数并创建新列
    df_traj['traj_index'] = df_traj['traj'].apply(lambda traj_list: [uli_index_dict.get(str(uli), uli) for uli in traj_list])

    build_graph_from_txt(df_traj, operator)
        
def build_graph_from_txt(traj_df, operator):
    # 初始化有向图
    G = nx.DiGraph()
    all_nodes = set()

    # 第一步: 收集所有节点
    for index, row in traj_df.iterrows():
        traj_index = row['traj_index']
        for i in range(len(traj_index)):
            u = traj_index[i]
            all_nodes.add(u)

    # 第二步: 将所有节点添加到图中
    for node in all_nodes:
        G.add_node(node)

    # 第三步: 根据文件内容添加边
    for index, row in traj_df.iterrows():
        traj_index = row['traj_index']
        for i in range(len(traj_index) - 1):
            u, v = traj_index[i], traj_index[i + 1]
            # 检查图中是否已经存在从 u 到 v 的边
            if G.has_edge(u, v):
                # 如果边已经存在, 增加边的权重
                G.edges[u, v]['weight'] += 1
            else:
                # 如果边不存在, 添加新边, 权重初始化为 1
                G.add_edge(u, v, weight=1)

    for node in G.nodes:
        G.add_edge(node, node, weight=1)

    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(f'../data/{operator}/graph/graph_A.csv', A.todense(), delimiter=',')

# gen_all_handle



# def gen_all(df_traj, operator):
#     # 读取uli_index_dict.json数据
#     with open(f'../data/{operator}/extra_file/uli_index_dict.json', 'r', encoding='utf-8') as file:
#         uli_index_dict = json.load(file)
    
#     result_dict = {}
#     timestamp_dict = {}
#     result_list = [0] * 200
#     timestamp_list = [0] * 200

#     for _, row in df_traj.iterrows():
#         trajlen = len(row['traj'])

#         # 映射轨迹点
#         mapped_traj = [uli_index_dict.get(str(uli), uli) for uli in row['traj']]
        
#         # 映射时间戳为小时数（0–23）
#         timestamp_list_raw = row.get('timestamp', [])
#         mapped_timestamp = [time.localtime(int(ts)).tm_hour for ts in timestamp_list_raw if ts]

#         # 存储轨迹映射结果
#         result_dict.setdefault(trajlen, []).append(mapped_traj)
#         timestamp_dict.setdefault(trajlen, []).append(mapped_timestamp)

#     # 构造统计数组（轨迹数量分布）
#     for key, value in result_dict.items():
#         if key < len(result_list):
#             result_list[key] = len(value)
#     for key, value in timestamp_dict.items():
#         if key < 200:
#             timestamp_list[key] = len(value)

#     result_array = np.array(result_list)
#     timestamp_array = np.array(timestamp_list)

#     os.makedirs(f'../data/{operator}/gen_all', exist_ok=True)

#     # 保存轨迹序列
#     with open(f'../data/{operator}/gen_all/index_seqs.bin', 'wb') as file:
#         pickle.dump(result_dict, file)
#     np.save(f'../data/{operator}/gen_all/length_distri', result_array)

#     # 保存时间戳序列
#     with open(f'../data/{operator}/gen_all/timestamp_seqs.bin', 'wb') as file:
#         pickle.dump(timestamp_dict, file)

#     # 保存时间戳为文本文件
#     with open(f'../data/{operator}/gen_all/timestamp_seqs.txt', 'w', encoding='utf-8') as txt_file:
#         txt_file.write(f"# 轨迹时间戳分布文件 - 运营商: {operator}\n")
#         txt_file.write(f"# 格式: 轨迹长度 | 转换后的小时数(0-23)\n")

#         for traj_len in sorted(timestamp_dict.keys()):
#             trajectories = timestamp_dict[traj_len]
#             txt_file.write(f"\n# 轨迹长度: {traj_len} (共{len(trajectories)}条轨迹)\n")
#             for ts_list in trajectories:
#                 txt_file.write(f"{' '.join(map(str, ts_list))}\n")

#     return result_dict, timestamp_dict

def gen_all(df_traj, operator):
    # 读取 uli -> index 映射表
    with open(f'../data/{operator}/extra_file/uli_index_dict.json', 'r', encoding='utf-8') as file:
        uli_index_dict = json.load(file)

    result_dict = {}   # 存储映射后的基站索引序列
    tim_dict = {}      # 存储每条轨迹的 tim 列（分钟）
    sta_dict = {}      # 存储每条轨迹的 sta 列（相邻点间的等待时间）

    # 1. 获取最大轨迹长度
    max_traj_len = df_traj['traj'].apply(len).max()
    
    # 根据最长轨迹长度初始化列表
    result_list = [0] * (max_traj_len + 1)
    tim_list = [0] * (max_traj_len + 1)
    sta_list = [0] * (max_traj_len + 1)


    for _, row in df_traj.iterrows():
        traj = row['traj']
        tim = row.get('tim', [])
        sta = row.get('sta', [])

        trajlen = len(traj)

        # 映射 uli → index
        mapped_traj = [uli_index_dict.get(str(uli), uli) for uli in traj]

        # 存入各个 dict 中
        result_dict.setdefault(trajlen, []).append(mapped_traj)
        tim_dict.setdefault(trajlen, []).append(tim)
        sta_dict.setdefault(trajlen, []).append(sta)

    # 构造长度分布数组
    for key in result_dict:
        if key < len(result_list):
            result_list[key] = len(result_dict[key])
            tim_list[key] = len(tim_dict.get(key, []))
            sta_list[key] = len(sta_dict.get(key, []))

    result_array = np.array(result_list)
    tim_array = np.array(tim_list)
    sta_array = np.array(sta_list)

    os.makedirs(f'../data/{operator}/gen_all', exist_ok=True)

    # 保存 index 序列
    with open(f'../data/{operator}/gen_all/index_seqs.bin', 'wb') as file:
        pickle.dump(result_dict, file)
    np.save(f'../data/{operator}/gen_all/length_distri.npy', result_array)

    # 保存 tim 序列
    with open(f'../data/{operator}/gen_all/tim_seqs.bin', 'wb') as file:
        pickle.dump(tim_dict, file)

    # 保存 sta 序列
    with open(f'../data/{operator}/gen_all/sta_seqs.bin', 'wb') as file:
        pickle.dump(sta_dict, file)

    return result_dict, tim_dict, sta_dict
    
    
# 做n2v、vgae嵌入
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
    # 进行one-hot编码
    df_bs_one_hot = pd.get_dummies(df_bs['c_lac'], prefix='c_lac')

    # 将编码后的数据合并到原数据框中
    df_bs = pd.concat([df_bs, df_bs_one_hot], axis=1)
    
    # 将DataFrame转换为numpy数组
    lac_one_hot_vector = df_bs_one_hot.to_numpy()
    return lac_one_hot_vector

def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    repo_root = Path(__file__).resolve().parents[3]
    split_root = repo_root / "privacy_eval" / "local" / "splits" / "SignalingTraj"

    embed_size = 64
    min_length, max_length = 1, 12000
    trajectory_parts = []
    station_parts = []

    for operator in ("0", "1", "3"):
        split_dir = split_root / f"operator{operator}"
        train_path = split_dir / "train.csv"
        holdout_path = split_dir / "holdout.csv"
        station_path = split_dir / "base_station_train.csv"
        if not train_path.is_file() or not holdout_path.is_file() or not station_path.is_file():
            raise FileNotFoundError(
                f"Missing split files for operator{operator}; run privacy_eval/prepare_all_splits.sh first"
            )
        trajectory_parts.append(pd.read_csv(train_path, dtype={"c_uli": str}))
        station_parts.append(pd.read_csv(station_path, dtype={"c_uli": str}))

    df_train = pd.concat(trajectory_parts, ignore_index=True)
    df_stations = pd.concat(station_parts, ignore_index=True)
    df_stations = df_stations.drop_duplicates(subset=["c_uli"]).reset_index(drop=True)

    df_0, df_1, df_3, bs_0, bs_1, bs_3 = traj_station_preprocess(
        df_train, df_stations, min_length, max_length
    )
    outputs = {
        "0": (df_0, bs_0),
        "1": (df_1, bs_1),
        "3": (df_3, bs_3),
    }

    prepared = 0
    for operator, (df_traj, df_bs) in outputs.items():
        tag = f"operator{operator}"
        data_root = Path("../data") / tag
        if data_root.exists():
            shutil.rmtree(data_root)

        if df_traj.empty or df_bs.empty:
            print(f"Skipping operator{operator}: no eligible training trajectories or stations")
            continue

        (data_root / "extra_file").mkdir(parents=True, exist_ok=True)
        (data_root / "graph").mkdir(parents=True, exist_ok=True)

        extra_file(df_bs, tag)
        create_graph(df_traj, tag)
        gen_all(df_traj, tag)
        gen_dir = data_root / "gen_all"
        train_length_distribution = np.load(gen_dir / "length_distri.npy")
        np.save(gen_dir / "length_distri_train.npy", train_length_distribution)

        holdout = pd.read_csv(holdout_path, dtype={"c_uli": str})
        holdout = holdout[holdout["c_uli"].isin(set(df_bs["c_uli"].astype(str)))]
        holdout_lengths = holdout.groupby(["c_number", "c_day"]).size()
        holdout_distribution = np.zeros(max_length + 1, dtype=np.int64)
        for length, count in holdout_lengths.items():
            if min_length <= length <= max_length:
                holdout_distribution[int(length)] = int(count)
        if holdout_distribution.sum() == 0:
            raise RuntimeError(
                f"No eligible holdout trajectories remain for operator{operator} "
                "after applying the training station vocabulary"
            )
        np.save(gen_dir / "length_distri_holdout.npy", holdout_distribution)
        np.save(gen_dir / "length_distri.npy", holdout_distribution)

        road_embed(df_traj, df_bs, embed_size, tag)
        prepared += 1

    if prepared == 0:
        raise RuntimeError("No operator artifacts were produced from the training split")


if __name__ == "__main__":
    main()
