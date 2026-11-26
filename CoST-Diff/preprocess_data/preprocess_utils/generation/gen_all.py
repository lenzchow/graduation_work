import pickle
import numpy as np
from preprocess.temporal_cond import (
    get_weekday_from_cday, build_slot_seq
)
import json

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
        mapped_traj = [uli_index_dict.get(str(uli), uli) for uli in traj]

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

    