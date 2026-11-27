import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch
import torch.optim as optim
import numpy as np
from utils.utils import save_json_data, create_dir
from utils.optimizer_utils import freeze_backbone, build_decoder_optimizer, build_full_optimizer
from torch.optim.lr_scheduler import StepLR

from models.model_utils import epoch_time, AttrDict
from models.multi_train import init_weights, train
from models.model import Diff_RNTraj
from models.diff_module import diff_CSDI
from build_graph import load_graph_adj_mtx
import warnings
import json
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
import random

FREEZE_EPOCH = 15
frozen = False

# 设置随机种子函数
# def set_seed(seed):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # cuDNN
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # 强制确定性算法
#     torch.use_deterministic_algorithms(False)
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 性能优先
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    

    
if __name__ == '__main__':
    # 设置随机种子
    set_seed(42)  # 可以选择任意整数作为种子

    parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
    parser.add_argument('--dataset', type=str, default='Chengdu',help='data set')
    parser.add_argument('--embed_mode', type=str, default='',help='embed mode')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=25, help='epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--diff_T', type=int, default=500, help='diffusion step')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='min beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='max beta')
    parser.add_argument('--pre_trained_dim', type=int, default=128, help='pre-trained dim of the road segment')
    parser.add_argument('--embedding_hour_dim', type=int, default=128, help='pre-trained dim of the embedding matrix')
    parser.add_argument('--rdcl', type=int, default=10, help='stack layers on the denoise network')
    parser.add_argument('--module_type', type=str, default='none',
                    choices=['none', 'lstm', 'transformer'])
    parser.add_argument('--tf_layers', type=int, default=2)
    parser.add_argument('--tf_nhead', type=int, default=4)
    parser.add_argument('--tf_dropout', type=float, default=0.1)


    # model params
    parser.add_argument('--use_fourier', type=bool, default=True, help='傅立叶编码类别')
    # model params
    parser.add_argument('--tim_dec_linear_hid1', type=int, default=256, help='时间解码器linear hid1')
    parser.add_argument('--tim_dec_linear_hid2', type=int, default=512, help='时间解码器linear hid2')
    
    opts = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert opts.dataset in ['shanghai', 'operator0','operator1','operator3'], 'Check dataset name if in [Porto, Chengdu]'

    print('Preparing data...')
    
    beta = np.linspace(opts.beta_start ** 0.5, opts.beta_end ** 0.5, opts.diff_T) ** 2
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    alpha = torch.tensor(alpha).float().to("cuda:0")
    alpha_bar = torch.tensor(alpha_bar).float().to("cuda:0")

    diffusion_hyperparams = {}
    diffusion_hyperparams['T'], diffusion_hyperparams['alpha_bar'], diffusion_hyperparams['alpha'] = opts.diff_T,  alpha_bar, alpha
    diffusion_hyperparams['beta'] = beta
    
    # test_flag = True
    test_flag = False

    path_dir = f'./data/{opts.dataset}/'

    extra_info_dir = path_dir + "extra_file/"

    UTG_file = path_dir + 'graph/graph_A.csv'
    if opts.dataset.startswith('operator'):
        if opts.embed_mode != '':
            pre_trained_road = path_dir + f'graph/road_embed_{opts.embed_mode}.txt'
        else:
            raise ValueError("embed_mode不能为空！")
    elif opts.dataset.startswith('shanghai'):
        if opts.embed_mode != '':
            pre_trained_road = path_dir + f'graph/road_embed_{opts.embed_mode}.txt'
        else:
            raise ValueError("embed_mode不能为空！")
    else:
        pre_trained_road = path_dir + 'graph/road_embed.txt'

    if test_flag:
        train_trajs_dir = path_dir + 'gen_debug/'
    else:
        train_trajs_dir = path_dir + 'gen_all/' 
                
                
    model_save_path = './results/'+opts.dataset + '/'
    create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a')
    # spatial embedding
    spatial_A = load_graph_adj_mtx(UTG_file)
    spatial_A_trans = np.zeros((spatial_A.shape[0], spatial_A.shape[1])) + 1e-10
    spatial_A_trans[:,:] = spatial_A

    f = open(pre_trained_road, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
    
    SE = torch.from_numpy(SE)

    args = AttrDict()

    if opts.dataset == 'operator0':
        args_dict = {
            'dataset': opts.dataset,

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':spatial_A_trans.shape[0],
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'embedding_hour_dim':opts.embedding_hour_dim,
            'use_fourier': opts.use_fourier,
            'tim_dec_linear_hid1':opts.tim_dec_linear_hid1,
            'tim_dec_linear_hid2':opts.tim_dec_linear_hid2,
            'rdcl': opts.rdcl,
            'module_type': opts.module_type,
            'tf_layers':opts.tf_layers,
            'tf_nhead':opts.tf_nhead,
            'tf_dropout':opts.tf_dropout
        }
    elif opts.dataset == 'operator1':
        args_dict = {
            'dataset': opts.dataset,

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':spatial_A_trans.shape[0],
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'embedding_hour_dim':opts.embedding_hour_dim,
            'use_fourier': opts.use_fourier,
            'tim_dec_linear_hid1':opts.tim_dec_linear_hid1,
            'tim_dec_linear_hid2':opts.tim_dec_linear_hid2,
            'rdcl': opts.rdcl,
            'module_type': opts.module_type,
            'tf_layers':opts.tf_layers,
            'tf_nhead':opts.tf_nhead,
            'tf_dropout':opts.tf_dropout
        }
    elif opts.dataset == 'operator3':
        args_dict = {
            'dataset': opts.dataset,

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':spatial_A_trans.shape[0],
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'embedding_hour_dim':opts.embedding_hour_dim,
            'use_fourier': opts.use_fourier,
            'tim_dec_linear_hid1':opts.tim_dec_linear_hid1,
            'tim_dec_linear_hid2':opts.tim_dec_linear_hid2,
            'rdcl': opts.rdcl,
            'module_type': opts.module_type,
            'tf_layers':opts.tf_layers,
            'tf_nhead':opts.tf_nhead,
            'tf_dropout':opts.tf_dropout
        }
    elif opts.dataset == 'shanghai':
        args_dict = {
            'dataset': opts.dataset,

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':spatial_A_trans.shape[0],
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'embedding_hour_dim':opts.embedding_hour_dim,
            'use_fourier': opts.use_fourier,
            'tim_dec_linear_hid1':opts.tim_dec_linear_hid1,
            'tim_dec_linear_hid2':opts.tim_dec_linear_hid2,
            'rdcl': opts.rdcl,
            'tf_layers':opts.tf_layers,
            'tf_nhead':opts.tf_nhead,
            'tf_dropout':opts.tf_dropout
        }

    args.update(args_dict)
    print(args)

    # load base_station index dataset   
    with open(train_trajs_dir + 'index_seqs.bin', 'rb') as f:  #路段序列
        all_index_seqs = pickle.load(f)
        f.close()
    
    # load base_station index dataset   
    with open(train_trajs_dir + 'tim_seqs.bin', 'rb') as f:  #路段序列
        all_tim_seqs = pickle.load(f)
        f.close()
    
    # load base_station index dataset   
    with open(train_trajs_dir + 'sta_seqs.bin', 'rb') as f:  #路段序列
        all_sta_seqs = pickle.load(f)
        f.close()
    
    with open(train_trajs_dir + 'slot_seqs.bin', 'rb') as f:
        all_slot_seqs = pickle.load(f)   # {L: List[List[int]]}
        f.close()
    with open(train_trajs_dir + 'seq_cond.bin', 'rb') as f:
        all_seq_cond = pickle.load(f)    # {L: List[int]}
        f.close()
        
    # 加载条件映射文件
    cond_feat = torch.from_numpy(np.load(extra_info_dir+'bs_feat.npy')).to(device)   # [N_bs, F] # lng_z','lat_z','ac_id','district_id'
    
    # cond_feat: [N,F], 列顺序 [lng_z, lat_z, ac_id, district_id]
    ac  = cond_feat[:, 2].to(torch.long)
    dist= cond_feat[:, 3].to(torch.long)

    # 去重个数
    n_ac   = int(ac.max().item()) + 1
    n_dist = int(dist.max().item()) + 1
    n_ac_no_unk   = ac[ac != 0].unique().numel()
    n_dist_no_unk = dist[dist != 0].unique().numel()
    
    # 展平所有等待时间
    dt = []
    for k, seqs in all_sta_seqs.items():          # seqs: 二维list
        for arr in seqs:                      # arr: 一条轨迹的等待序列
            dt.extend(arr)

    dt = np.asarray(dt, dtype=float)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]                           # 过滤占位/无效
    
    # t_min   = max(0.3, float(np.percentile(dt, 1)))
    t_min = 0.3
    U_head  = float(np.percentile(dt, 99.5))   # 仅供 head 映射
    U_nll   = U_head                              # 训练不截断
    mean_dt = float(dt.mean())
    mu_train = float(np.log(1.0 / np.clip(dt, t_min, U_nll)).mean())

    
    diff_model = diff_CSDI(args.hid_dim, args.hid_dim, opts.diff_T, args.hid_dim, args.rdcl, args)
    model = Diff_RNTraj(diff_model, diffusion_hyperparams, args, n_ac=n_ac, n_dist=n_dist,device=device).to(device)
    # print(model,'model')
    model.apply(init_weights)                         # 1) 先初始化
    
    logging.info('\n\n')
    open(model_save_path+'logging.txt', 'w').close()
        
    ls_train_loss, ls_train_const_loss, ls_train_diff_loss, ls_train_x0_loss, ls_train_sta_loss = [], [], [], [], []
    
    dict_train_loss = {}
    
    best_loss = float('inf')  # compare id loss

    # get all parameters (model parameters + task dependent log variances)
    log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param

    optimizer, scheduler = build_full_optimizer(
        model,
        base_lr=args.learning_rate,
        wd=1e-4
    )

    for epoch in tqdm(range(args.n_epochs)):
        start_time = time.time()

        # ===== Stage switch =====
        if epoch == FREEZE_EPOCH and not frozen:
            print(f"[INFO] Switch to decoder-only training at epoch {epoch}")

            freeze_backbone(model)

            # 固定 log_vars（非常重要）
            for v in log_vars:
                v.requires_grad = False

            optimizer, scheduler = build_decoder_optimizer(
                model,
                base_lr=args.learning_rate
            )

            frozen = True

        new_log_vars, train_loss, train_const_loss, train_diff_loss, train_x0_loss, train_sta_loss = \
                train(model, spatial_A_trans, 
                    SE, 
                    all_tim_seqs,
                    all_sta_seqs, 
                    all_index_seqs, 
                    all_slot_seqs,
                    all_seq_cond,
                    optimizer, 
                    log_vars, 
                    epoch,
                    args, cond_feat, diffusion_hyperparams)
        scheduler.step()
        

        ls_train_loss.append(train_loss)
        ls_train_const_loss.append(train_const_loss)
        ls_train_diff_loss.append(train_diff_loss)
        ls_train_x0_loss.append(train_x0_loss)
        ls_train_sta_loss.append(train_sta_loss)
        

        dict_train_loss['train_ttl_loss'] = ls_train_loss
        dict_train_loss['train_const_loss'] = ls_train_const_loss
        dict_train_loss['train_diff_loss'] = ls_train_diff_loss
        dict_train_loss['train_x0_loss'] = ls_train_x0_loss
        dict_train_loss['train_sta_loss'] = ls_train_sta_loss

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if train_loss < best_loss:
            best_loss = train_loss
            print("saving.....")
            torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')


        # logging 是 log.txt
        logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
        weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
        logging.info('log_vars:' + str(weights))
        logging.info('\tTrain Loss:' + str(train_loss) +
                        '\tTrain Const Loss:' + str(train_const_loss) +
                        '\tTrain Diff Loss:' + str(train_diff_loss) +
                        '\tTrain X0 Loss:' + str(train_x0_loss)+
                        '\tTrain sta Loss:' + str(train_sta_loss)+
                        '\n'
                        )
        
        # logging.txt
        with open(model_save_path+'logging.txt', 'a+') as f:
            f.write('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
            f.write('\tTrain Loss:' + str(train_loss) +
                        '\tTrain Const Loss:' + str(train_const_loss) +
                        '\tTrain Diff Loss:' + str(train_diff_loss) +
                        '\tTrain X0 Loss:' + str(train_x0_loss) +
                        '\tTrain sta Loss:' + str(train_sta_loss)+
                        '\n')
            
        
        torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
        save_json_data(dict_train_loss, model_save_path, "train_loss.json")
    
    

