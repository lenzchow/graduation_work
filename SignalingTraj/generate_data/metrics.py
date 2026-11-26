import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats
from scipy.stats import entropy

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
        if (graph[point_t][point_t_plus_1] != 1e-10):
            numerator += 1
        count += 1
    rsc = (numerator / (T - 1)) * 100
    return rsc

# f_jsd
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
        # freq = np.array(freq, dtype=float) / trajs.shape[1]
        freq = np.array(freq, dtype=float) / len(traj)
        topk_visits_loc.append(loc)
        topk_visits_freq.append(freq)
    topk_visits_loc = np.array(topk_visits_loc, dtype=int)
    topk_visits_freq = np.array(topk_visits_freq, dtype=float)
    # print(topk_visits_loc)
    return topk_visits_loc, topk_visits_freq

def get_overall_topk_visits_freq(trajs, k):
        topk_visits_loc, topk_visits_freq = get_topk_visits_f(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn), topk_visits_loc

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

def get_periodicity(trajs):
    """
    stat how many repetitions within a single trajectory
    :param trajs:
    :return:
    """
    reps = []
    for traj in trajs:
        # 计算每个轨迹中不同元素的比例
        reps.append(float(len(set(traj))) / len(traj))
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
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
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
    js = 0.5 * scipy.stats.entropy(p1, m) + \
        0.5 * scipy.stats.entropy(p2, m)
    return js


def calculate_jsd(p, q):
    """
    计算两个概率分布之间的Jensen - Shannon散度
    :param p: 第一个概率分布
    :param q: 第二个概率分布
    :return: JSD值
    """
    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return jsd

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

def evaluation(method, data_all, data_test, data_gen, numPoint):
    '''
    method: 0:arr_to_distribution使用默认参数0,1; 1:通过get_overall_topk_visits_freq使用动态参数
    numPoint: 为基站个数
    三个data文件:格式为[[1,2,3],[1,2,2,2],[1,3,4]]
    '''

    f1,loc1 = get_overall_topk_visits_freq(data_gen, 100)
    f2,loc2 = get_overall_topk_visits_freq(data_all, 100)

    if method == 0:
        min_val = 0
        max_val = 1
    elif method == 1:
        min_val = min([min(f1),min(f2)])
        max_val = max([max(f1),max(f2)])

    res = {}

    # Dailyloc
    test_reps = get_periodicity(data_all)
    gene_reps = get_periodicity(data_gen)
    test_dist, _ = arr_to_distribution(test_reps, 0, 1, 10)
    gene_dist, _ = arr_to_distribution(gene_reps, 0, 1, 10)
    # test_dist, _ = arr_to_distribution(test_reps,min([min(f1),min(f2)]),max([max(f1),max(f2)]),100)
    # gene_dist, _ = arr_to_distribution(gene_reps,min([min(f1),min(f2)]),max([max(f1),max(f2)]),100)
    p_jsd = get_js_divergence(test_dist, gene_dist) # 每个用户每天访问的位置数
    print("Dailyloc",p_jsd)
    res["Dailyloc"] = p_jsd

    # G-rank
    f1,loc1 = get_overall_topk_visits_freq(data_gen, 100)
    f2,loc2 = get_overall_topk_visits_freq(data_all, 100)
    l1 =  get_visits(data_all,numPoint)
    l2 =  get_visits(data_gen,numPoint)
    l1_dist, _ = get_topk_visits_l(l1, 100)
    l2_dist, _ = get_topk_visits_l(l2, 100)
    l1_dist_distribution, _ = arr_to_distribution(l1_dist,min_val,max_val,100)
    l2_dist_distribution, _ = arr_to_distribution(l2_dist,min_val,max_val,100)
    l_jsd = get_js_divergence(l1_dist_distribution, l2_dist_distribution) # 前100名地点的访问频率
    print("G-rank",l_jsd)
    res["G-rank"] = l_jsd

     # I-rank
    # f1,loc1 = get_overall_topk_visits_freq(data_gen, 100)
    # f2,loc2 = get_overall_topk_visits_freq(data_all, 100)
    f1_dist, _ = arr_to_distribution(f1,min_val,max_val,100)
    f2_dist, _ = arr_to_distribution(f2,min_val,max_val,100)
    f_jsd = get_js_divergence(f1_dist, f2_dist) # 个人版本的-前100名地点的访问频率
    print('I-rank',f_jsd)
    res["I-rank"] = f_jsd

    # Areacode


    # 示例数据，这里简单划分前两条为真实数据，最后一条为生成数据
    real_data = data_test
    generated_data = data_gen

    # 计算基站分布概率
    unique_real, probs_real = get_base_station_distribution(real_data)
    unique_generated, probs_generated = get_base_station_distribution(generated_data)

    # 对齐概率分布
    p_aligned, q_aligned = align_distributions(unique_real, probs_real, unique_generated, probs_generated)

    # 计算JSD
    jsd_value = calculate_jsd(p_aligned, q_aligned)

    print("Areacode", jsd_value)
    res["Areacode"] = jsd_value

    print(res)