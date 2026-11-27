import numpy as np
import random
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from models.model_utils import touli, sample_arrival_times, compute_global_bs_frequency
from models.loss_fn import cal_id_acc, cal_id_acc_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('multi_task device', device)

def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)

def next_batch(time, sta, indexes, slot, seqcond):
    length = len(indexes)
    for i in range(length):
        yield time[i], sta[i], indexes[i], slot[i], seqcond[i]
        
def train(model, spatial_A_trans, SE,
          all_tim_seqs, all_sta_seqs, all_index_seqs,
          all_slot_seqs, all_seq_cond,
          optimizer, log_vars, epoch,parameters, cond_feat, diffusion_hyperparams):

    criterion_reg  = nn.MSELoss()
    criterion_ce   = nn.NLLLoss()
    criterion_ce1  = nn.NLLLoss()
    epoch_time_hour_loss = 0
    epoch_ttl_loss  = 0
    epoch_const_loss = 0
    epoch_diff_loss  = 0
    epoch_x0_loss    = 0
    epoch_sta_loss   = 0

    # ===== 构造 batch 列表（按长度分桶） =====
    all_length = []
    all_batch_time, all_batch_sta, all_batch_index = [], [], []
    all_batch_slot, all_batch_seqcond = [], []

    # 计算基站在所有轨迹中出现的频次
    global_bs_freq, global_bs_count = compute_global_bs_frequency(
        all_index_seqs,
        num_bs=SE.shape[0]
    )
    global_bs_freq = global_bs_freq.to(device)

    for k, eids_group in all_index_seqs.items():
        all_length.append(k)
        traj_num = len(eids_group) // parameters.batch_size

        # 整批
        for i in range(traj_num):
            s = i * parameters.batch_size
            e = (i + 1) * parameters.batch_size

            all_batch_time.append(all_tim_seqs[k][s:e])
            all_batch_sta.append(all_sta_seqs[k][s:e])
            all_batch_index.append(all_index_seqs[k][s:e])

            all_batch_slot.append(all_slot_seqs[k][s:e])        # 事件条件 [B,L]
            all_batch_seqcond.append(all_seq_cond[k][s:e])      # 序列条件 [B]

        # 尾巴一小批（不足 batch_size）
        if traj_num * parameters.batch_size != len(eids_group):
            s = traj_num * parameters.batch_size

            all_batch_time.append(all_tim_seqs[k][s:])
            all_batch_sta.append(all_sta_seqs[k][s:])
            all_batch_index.append(all_index_seqs[k][s:])

            all_batch_slot.append(all_slot_seqs[k][s:])
            all_batch_seqcond.append(all_seq_cond[k][s:])

    # ===== 打乱所有 batch（保持对齐） =====
    zipped_list = list(zip(all_batch_time,
                           all_batch_sta,
                           all_batch_index,
                           all_batch_slot,
                           all_batch_seqcond))
    random.shuffle(zipped_list)
    (all_batch_time,
     all_batch_sta,
     all_batch_index,
     all_batch_slot,
     all_batch_seqcond) = zip(*zipped_list)

    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)

    cnt = 0
    for tim, sta, indexs, slot, seqcond in next_batch(
            all_batch_time,
            all_batch_sta,
            all_batch_index,
            all_batch_slot,
            all_batch_seqcond):

        cnt += 1
        import time

        # [B,L]
        src_time_seqs  = torch.tensor(tim,    dtype=torch.float32, device=device)
        src_sta_seqs   = torch.tensor(sta,    dtype=torch.float32, device=device)
        src_index_seqs = torch.tensor(indexs, dtype=torch.long,    device=device)
        src_slot_seqs  = torch.tensor(slot,   dtype=torch.long,    device=device)

        # [B] → [B,1]
        src_seq_cond = torch.tensor(seqcond,  dtype=torch.long, device=device).unsqueeze(-1)

        optimizer.zero_grad()
        curr_time = time.time()

        diff_loss, const_loss, x0_loss, sta_loss, freq_kl = model(
            spatial_A_trans, SE,
            src_sta_seqs,
            src_index_seqs,
            src_time_seqs,
            cond_feat,
            src_slot_seqs,
            src_seq_cond,
            real_freq = global_bs_freq
        )
        if epoch < 15:
            ttl_loss = diff_loss + const_loss + x0_loss + 1.0 * sta_loss + 0.5 * freq_kl
        else:
            ttl_loss = const_loss + sta_loss + 0.5 * freq_kl

        curr_time = time.time()
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)
        optimizer.step()

        epoch_ttl_loss   += ttl_loss.item()
        epoch_const_loss += const_loss.item()
        epoch_diff_loss  += diff_loss.item()
        epoch_x0_loss    += x0_loss.item()
        epoch_sta_loss   += sta_loss.item()
        # epoch_sta_loss   += sta_loss.item()

    return (log_vars,
            epoch_ttl_loss / cnt,
            epoch_const_loss / cnt,
            epoch_diff_loss / cnt,
            epoch_x0_loss / cnt,
            epoch_sta_loss / cnt)


def generate_data(model, 
                  spatial_A_trans, 
                  index_uli_dict, 
                  index_lat_lon_dict, 
                  index_areacode_dict, 
                  args, SE, formatted_time, 
                  t_min, U_business, cond_feat, 
                  all_index_seqs, all_tim_seqs, all_slot_seqs, all_seq_cond):
    
    model.eval()  
    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
    
    if args.dataset.startswith("operator") or args.dataset.startswith("shanghai"):
        length2num = np.load(f"./data/{args.dataset}/gen_all/length_distri.npy")

        # 找到不为0的元素的下标
        non_zero_indices = np.nonzero(length2num)[0]

        start_length = non_zero_indices[0] if non_zero_indices.size > 0 else None
        end_length = non_zero_indices[-1] if non_zero_indices.size > 0 else None
        
    traj_all_num = int(length2num.sum())

    with torch.no_grad():  # this line can help speed up evaluation
        for i in tqdm(range(start_length, end_length+1)):
            
            curr_length =i
            curr_batch = int(length2num[i] / length2num.sum() * traj_all_num)

            while True:
                if curr_batch > 256:
                    _curr_batch = 256
                else:
                    _curr_batch = curr_batch

                sta, ids = model.generate_data(spatial_A_trans, SE, _curr_batch, curr_length, 
                                               args.pre_trained_dim, t_min, U_business, cond_feat, all_index_seqs, all_tim_seqs,all_slot_seqs, all_seq_cond)

                arrival_time = sample_arrival_times(model, ids, sta)        

                # —— 预算内截断：在首个越界处截断，后续清零 ——
                # arrival_time: [B, T]；sta: [B, T-1]
                budget = 1440.0
                B, T = arrival_time.shape

                # bool 矩阵：到达时间是否越界
                over_all = arrival_time > budget
                has_over = over_all.any(dim=1)            # [B]

                if has_over.any():
                    b_idx = torch.nonzero(has_over, as_tuple=False).squeeze(1)      # 哪些轨迹越界
                    first_over = torch.argmax(over_all[has_over].to(torch.int64), dim=1)  # 每条的首个越界下标 k
                    prev_idx = (first_over - 1).clamp_min(0)                         # k 的前一段下标

                    # 取首个越界前的到达时间 t_{k-1}
                    prev_t = arrival_time[b_idx, prev_idx]
                    remain = (budget - prev_t).clamp(min=0.0)

                    # 把首个越界点的到达时间截到 1440，并把对应等待时长改为 remain
                    arrival_time[b_idx, first_over] = prev_t + remain
                    sta[b_idx, prev_idx] = remain

                    # 首个越界点之后的到达时间设为截断值，等待时长清零（等价于“截断”）
                    for bi, k, pi in zip(b_idx.tolist(), first_over.tolist(), prev_idx.tolist()):
                        if k + 1 < T:
                            arrival_time[bi, k+1:] = arrival_time[bi, k]
                        if pi + 1 < T - 1:
                            sta[bi, pi+1:] = 0.0

                # 直接写整批（若写出逻辑会把 sta==0 的尾部也写出，可在写文件处依据 sta>0 截断）
                touli(index_uli_dict,
                      index_lat_lon_dict,
                      index_areacode_dict,
                      arrival_time,
                      sta,
                      ids,
                      args,
                      start_length,
                      end_length,
                      formatted_time,
                      save_txt_num=i)

                # 4. 更新剩余要生成的数量
                curr_batch = curr_batch - _curr_batch
                
                if curr_batch <= 0: break