import numpy as np
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from models.model_utils import touli, get_constraint_mask
from models.loss_fn import cal_id_acc, check_rn_dis_loss, cal_id_acc_train
from models.trajectory_graph import build_graph,search_road_index

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

def next_batch(timestamp_hour, indexes):
    length = len(indexes)
    for i in range(length):
        yield timestamp_hour[i], indexes[i]
        
def train(model, spatial_A_trans, SE, all_timestamp_seqs,all_index_seqs, optimizer, log_vars, parameters, diffusion_hyperparams):
# def train(model, spatial_A_trans, SE, all_eids, all_rates, optimizer, log_vars, parameters, diffusion_hyperparams):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()
    criterion_ce1 = nn.NLLLoss()
    epoch_time_hour_loss = 0
    epoch_ttl_loss = 0
    epoch_const_loss = 0
    epoch_diff_loss = 0
    epoch_x0_loss = 0
    

    # timestamp
    all_length = []
    all_batch_timestmap, all_batch_index = [],[]
    for k, eids_group in all_index_seqs.items():
        all_length.append(k)
        traj_num = len(eids_group) // parameters.batch_size
        for i in range(traj_num):
            all_batch_timestmap.append(all_timestamp_seqs[k][i*parameters.batch_size:(i+1)*parameters.batch_size])
            all_batch_index.append(all_index_seqs[k][i*parameters.batch_size:(i+1)*parameters.batch_size])
        if traj_num * parameters.batch_size != len(eids_group):
            all_batch_timestmap.append(all_timestamp_seqs[k][traj_num*parameters.batch_size:])
            all_batch_index.append(all_index_seqs[k][traj_num*parameters.batch_size:])
    
    # 打乱数据
    zipped = zip(all_batch_timestmap, all_batch_index)
    zipped_list = list(zipped)
    random.shuffle(zipped_list)
    all_batch_timestmap, all_batch_index = zip(*zipped_list)
    


    SE = SE.to(device)
    
    cnt = 0
    for timestamps, indexs in next_batch(all_batch_timestmap, all_batch_index):
    # for indexs in next_batch(all_batch_index):
        cnt += 1
        import time
        
        src_timestamp_seqs = torch.tensor(timestamps).long().to(device)
        src_index_seqs = torch.tensor(indexs).long().to(device)
        
        spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
        curr_time = time.time()
        optimizer.zero_grad()
        diff_loss, const_loss, x0_loss, time_hour_loss, time_hour_embedding = model(spatial_A_trans, SE, src_timestamp_seqs, src_index_seqs)  # T, B, id_size   and    T, B, 1
        
        ttl_loss = diff_loss + const_loss + x0_loss + time_hour_loss
        # ttl_loss = diff_loss + x0_loss 

        curr_time = time.time()
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()
        
        
        epoch_ttl_loss += ttl_loss.item()
        epoch_const_loss += const_loss.item()
        epoch_diff_loss += diff_loss.item()
        epoch_x0_loss += x0_loss.item()
        epoch_time_hour_loss += time_hour_loss.item() # ? 有没有.item()
        
    return log_vars, epoch_ttl_loss / cnt, epoch_const_loss / cnt, epoch_diff_loss / cnt, epoch_x0_loss / cnt, epoch_time_hour_loss / cnt, time_hour_embedding


def generate_data(model, 
                  spatial_A_trans, 
                  index_uli_dict, 
                  index_lat_lon_dict, 
                  index_areacode_dict, 
                  time_embedding_hour_matrix,
                  parameters, SE, formatted_time):
    model.eval()  
    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
    
    if parameters.dataset.startswith("operator"):
        length2num = np.load(f"./data/{parameters.dataset}/gen_all/length_distri.npy")

        # 找到不为0的元素的下标
        non_zero_indices = np.nonzero(length2num)[0]

        start_length = non_zero_indices[0] if non_zero_indices.size > 0 else None
        end_length = non_zero_indices[-1] if non_zero_indices.size > 0 else None
    # batchsize=1
    traj_all_num = int(length2num.sum())
    # batchsize=1
    with torch.no_grad():  # this line can help speed up evaluation
        for i in tqdm(range(start_length, end_length+1)):
            curr_length =i
            curr_batch = int(length2num[i] / length2num.sum() * traj_all_num)

            while True:
                if curr_batch > 256: 
                    _curr_batch = 256
                else:
                    _curr_batch = curr_batch
                times, ids = model.generate_data(spatial_A_trans, SE, _curr_batch, curr_length, parameters.pre_trained_dim, time_embedding_hour_matrix)

                output_seqs = touli(index_uli_dict, 
                                    index_lat_lon_dict ,
                                    index_areacode_dict,
                                    times, ids, 
                                    parameters, 
                                    start_length,end_length,
                                    formatted_time,
                                    save_txt_num = i
                                    )
                curr_batch = curr_batch - _curr_batch
                if curr_batch <= 0: break
                # exit()
            # print(output_seqs)
        # exit()

