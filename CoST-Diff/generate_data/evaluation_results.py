import os
import json
import pickle
import numpy as np
import pandas as pd
import glob
from metric import (
    load_graph_adj_mtx, calculate_single_rsc, arr_to_distribution, get_js_divergence,
    get_topk_visits_f, get_overall_topk_visits_freq, get_visits, get_topk_visits_l,
    get_periodicity, calculate_jsd_for_area_codes, get_base_station_distribution,
    align_distributions, calculate_jsd
)

def load_generated_data(directory):
    """读取生成的轨迹数据"""
    gene_data = []
    gene_data_coordinate = []
    gene_data_areacode = []
    gene_data_time_hour = []
    gene_data_sta_tim = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            current_areacode_list = []
            current_index_list = []
            current_coordinate_list = []
            current_time_hour_list = []
            current_sta_time_list = []
            with open(file_path) as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("#"):
                        if current_index_list:
                            gene_data.append(current_index_list.copy())
                            current_index_list = []
                        if current_coordinate_list:
                            gene_data_coordinate.append(current_coordinate_list.copy())
                            current_coordinate_list = []
                        if current_time_hour_list:
                            gene_data_time_hour.append(current_time_hour_list.copy())
                            current_time_hour_list = []
                        if current_sta_time_list:
                            gene_data_sta_tim.append(current_sta_time_list.copy())
                            current_sta_time_list = []
                    else:
                        parts = line.split(',')
                        areacode = int(parts[0])
                        index = int(parts[3])
                        longitude = float(parts[1].strip())
                        latitude = float(parts[2].strip())
                        time_hour = float(parts[4])
                        sta_time = float(parts[5])

                        current_areacode_list.append(areacode)
                        current_index_list.append(index)
                        current_coordinate_list.append((longitude, latitude))
                        current_time_hour_list.append(time_hour)
                        current_sta_time_list.append(sta_time)

                # 处理最后一条轨迹
                if current_areacode_list:
                    gene_data_areacode.append(current_areacode_list)
                if current_index_list:
                    gene_data.append(current_index_list)
                if current_coordinate_list:
                    gene_data_coordinate.append(current_coordinate_list)
                if current_time_hour_list:
                    gene_data_time_hour.append(current_time_hour_list)
                if current_sta_time_list:
                    gene_data_sta_tim.append(current_sta_time_list)

    return gene_data, gene_data_coordinate, gene_data_areacode, gene_data_time_hour, gene_data_sta_tim


def load_real_data(operator):
    
    """读取真实轨迹数据并映射areacode"""
    with open(f'./data/{operator}/gen_all/index_seqs.bin', 'rb') as f:
        data_dict = pickle.load(f)
        
    with open(f'./data/{operator}/gen_all/sta_seqs.bin', 'rb') as f:
        test_data_sta_tim = json.load(f)
        
    with open(f'./data/{operator}/gen_all/tim_seqs.bin', 'rb') as f:
        test_data_time_hour = json.load(f)

    with open(f'./data/{operator}/extra_file/index_areacode_dict.json', 'r', encoding='utf-8') as f:
        index_areacode_dict = json.load(f)


    test_data = []
    for sub_list in data_dict.values():
        test_data.extend(sub_list)

    test_areacode_data = []
    for sub_list in test_data:
        mapped_sub_list = [index_areacode_dict.get(str(element), element) for element in sub_list]
        test_areacode_data.append(mapped_sub_list)

    return test_data, test_areacode_data, test_data_time_hour, test_data_sta_tim

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
    # 将到达时间按分钟slot分组，统计访问次数
    def time_to_slot_counts(tim_list):
        counts = np.zeros(1440 // slot)
        for traj in tim_list:
            slots = (np.array(traj) % 1440) // slot
            for s in slots:
                counts[int(s)] += 1
        return counts

    real_counts = time_to_slot_counts(real_tim)
    gen_counts = time_to_slot_counts(gen_tim)

    p_real = real_counts / np.sum(real_counts)
    p_gen = gen_counts / np.sum(gen_counts)

    return JSD(p_gen, p_real)

# --------- Stay ----------
def calc_stay_mse(real_tim, real_sta, gen_tim, gen_sta, slot=60):
    # 每个slot平均等待时间
    def slot_avg_wait(tim_list, sta_list):
        wait_times = [[] for _ in range(1440 // slot)]
        for traj_tim, traj_sta in zip(tim_list, sta_list):
            slots = (np.array(traj_tim) % 1440) // slot
            for s, w in zip(slots, traj_sta):
                wait_times[int(s)].append(w)
        avg_wait = np.array([np.mean(w) if len(w) > 0 else 0 for w in wait_times])
        return avg_wait

    avg_real = slot_avg_wait(real_tim, real_sta)
    avg_gen = slot_avg_wait(gen_tim, gen_sta)

    mse = np.mean((avg_real - avg_gen) ** 2)
    return mse


def evaluate_metrics(
    gene_data, gene_data_areacode, gene_data_time_hour, gene_data_sta_tim,
    test_data, test_areacode_data, test_data_time_hour, test_data_sta_tim,
    G, shape, directory,
    csv_path="autodl-tmp/signalingTraj_sep/generate_data/result_evaluation.csv"
):
    """计算一系列指标并将结果追加保存到CSV，第一次运行固定列顺序"""

    # ===== 计算各指标 =====
    results = {}

    # -------- 空间角度指标 --------
    # I-rank
    f1 = get_overall_topk_visits_freq(test_data, 100)
    f2 = get_overall_topk_visits_freq(gene_data, 100)
    f1_dist, _ = arr_to_distribution(f1, 0, 1, 100)
    f2_dist, _ = arr_to_distribution(f2, 0, 1, 100)
    results["I-rank"] = get_js_divergence(f1_dist, f2_dist)

    # G-rank
    l1 = get_visits(test_data, shape[0])
    l2 = get_visits(gene_data, shape[0])
    l1_dist, _ = get_topk_visits_l(l1, 100)
    l2_dist, _ = get_topk_visits_l(l2, 100)
    l1_dist_distribution, _ = arr_to_distribution(l1_dist, 0, 0.05, 100)
    l2_dist_distribution, _ = arr_to_distribution(l2_dist, 0, 0.05, 100)
    results["G-rank"] = get_js_divergence(l1_dist_distribution, l2_dist_distribution)

    # Dailyloc
    test_lengths = [len(traj) for traj in test_data]
    gene_lengths = [len(traj) for traj in gene_data]
    test_reps = get_periodicity(test_data, test_lengths)
    gene_reps = get_periodicity(gene_data, gene_lengths)
    test_dist, _ = arr_to_distribution(test_reps, 0, 1, 100)
    gene_dist, _ = arr_to_distribution(gene_reps, 0, 1, 100)
    results["Dailyloc"] = get_js_divergence(test_dist, gene_dist)

    # Average RSC
    all_rsc_values = [calculate_single_rsc(traj, G) for traj in gene_data]
    results["Average RSC"] = np.mean(all_rsc_values)

    # Base station JSD
    unique_real, probs_real = get_base_station_distribution(test_data)
    unique_generated, probs_generated = get_base_station_distribution(gene_data)
    p_aligned, q_aligned = align_distributions(unique_real, probs_real, unique_generated, probs_generated)
    results["Base station JSD"] = calculate_jsd(p_aligned, q_aligned)

    # Areacode JSD
    results["Areacode JSD"] = calculate_jsd_for_area_codes(test_areacode_data, gene_data_areacode)

    # -------- 时间角度指标 --------
    results["Duration"] = calc_duration_jsd(test_data_sta_tim, gene_data_sta_tim)
    results["Move"] = calc_move_jsd(test_data_time_hour, gene_data_time_hour, slot=10)
    results["Stay"] = calc_stay_mse(test_data_time_hour, test_data_sta_tim, gene_data_time_hour, gene_data_sta_tim, slot=10)

    # ===== 构建一行数据 =====
    dir_name = os.path.basename(os.path.normpath(directory))
    row_data = {"directory": dir_name}
    row_data.update(results)

    # ===== 写入CSV =====
    df = pd.DataFrame([row_data])

    # 固定列顺序（加上新指标）
    columns_order = ["directory", "I-rank", "G-rank", "Dailyloc",
                     "Duration", "Move", "Stay", 
                     "Average RSC", "Base station JSD", "Areacode JSD"
                     ]

    if not os.path.exists(csv_path):
        df = df[columns_order]
        df.to_csv(csv_path, index=False)
    else:
        df = df[columns_order]
        df.to_csv(csv_path, mode='a', header=False, index=False)

    return results

if __name__ == "__main__":
    operator = 'operator0'
    base_dir = './generate_data/'
    
    # 获取所有子目录
    subdirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]

    if not subdirs:
        raise FileNotFoundError(f"在 {base_dir} 中没有找到子目录")

    # 找最近修改时间的子目录
    directory = max(subdirs, key=os.path.getmtime)

    gene_data, gene_data_coordinate, gene_data_areacode, gene_data_time_hour, gene_data_sta_tim = load_generated_data(directory)
    test_data, test_areacode_data, test_data_time_hour, test_data_sta_tim = load_real_data(operator)
    G = load_graph_adj_mtx(f'./data/{operator}/graph/graph_A.csv')

    shape = G.shape  # 原来代码里用 shape[0]

    evaluate_metrics(
    gene_data, gene_data_areacode, gene_data_time_hour, gene_data_sta_tim,
    test_data, test_areacode_data, test_data_time_hour, test_data_sta_tim,
    G, shape, directory
)