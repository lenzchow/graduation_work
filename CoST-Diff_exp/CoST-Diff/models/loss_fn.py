import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.spatial_func import SPoint, distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_dis_loss(predict, target, trg_len):
    """
    Calculate MAE and RMSE between predicted and targeted GPS sequence.
    Args:
    -----
        predict = [seq len, batch size, 2]
        target = [seq len, batch size, 2]
        trg_len = [batch size]  if not considering target length, the loss will smaller than the real one.
        predict and target have been removed sos 
    Returns:
    -------
        MAE of a batch in meter.
        RMSE of a batch in meter.
    """
    predict = predict.permute(1, 0, 2)  # [batch size, seq len, 2]
    target = target.permute(1, 0, 2)  # [batch size, seq len, 2]
    bs = predict.size(0)

    ls_dis = []
    for bs_i in range(bs):
        for len_i in range(trg_len[bs_i]-1):
            pre = SPoint(predict[bs_i, len_i][0], predict[bs_i, len_i][1])
            trg = SPoint(target[bs_i, len_i][0], target[bs_i, len_i][1])
            dis = distance(pre, trg)
            ls_dis.append(dis)

    ls_dis = np.array(ls_dis)
    mae = ls_dis.mean()
    rmse = np.sqrt((ls_dis**2).mean())
    return mae, rmse


def shrink_seq(seq):
    """remove repeated ids"""
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s
    
    return new_seq

def memoize(fn):
    '''Return a memoized version of the input function.

    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function 
    is called repeatedly with the same arguments.
    '''
    cache = dict()
    def wrapped(*v):
        key = tuple(v) # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]
    return wrapped

def lcs(xs, ys):
    '''Return the longest subsequence common to xs and ys.

    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''
    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i-1], ys[j-1]
            if xe == ye:
                return lcs_(i-1, j-1) + [xe]
            else:
                return max(lcs_(i, j-1), lcs_(i-1, j), key=len)
        else:
            return []
    return lcs_(len(xs), len(ys))


def cal_id_acc(predict, target, trg_len):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len, batch size, id one hot output dim]
        target = [seq len, batch size, 1]
        predict and target have been removed sos 
    Returns:
    -------
        mean matched RID accuracy.
    """
    predict = predict.permute(1, 0, 2)  # [batch size, seq len, id dim]
    target = target.permute(1, 0)  # [batch size, seq len, 1]
    bs = predict.size(0)

    correct_id_num = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    ttl = 0
    cnt = 0
    for bs_i in range(bs):
        pre_ids = []
        trg_ids = []
        # -1 because predict and target are removed sos.
        for len_i in range(trg_len[bs_i] - 1):
            pre_id = predict[bs_i][len_i].argmax()
            trg_id = target[bs_i][len_i]
            pre_ids.append(pre_id)
            trg_ids.append(trg_id)
            if pre_id == trg_id:
                cnt += 1
            ttl += 1

        # compute average rid accuracy
        shr_trg_ids = shrink_seq(trg_ids)
        shr_pre_ids = shrink_seq(pre_ids)
        correct_id_num += len(lcs(shr_trg_ids, shr_pre_ids))
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)

    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    return rid_acc, rid_recall, rid_precision

def cal_id_acc_train(predict, target, test=False):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len, batch size, id one hot output dim]
        target = [seq len, batch size, 1]
        predict and target have been removed sos 
    Returns:
    -------
        mean matched RID accuracy.
    """
    predict = predict.permute(1, 0, 2)  # [batch size, seq len, id dim]
    target = target.permute(1, 0)  # [batch size, seq len, 1]
    bs = predict.size(0)

    correct_id_num = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    ttl = 0
    cnt = 0
    predict_id = predict.argmax(-1)
    # print(predict_id[0])
    # print(target[0])
    # exit()
    sum1 = predict_id.shape[0] * predict_id.shape[1]
    # sum2 = torch.sum(target==0)
    # mask = (target != 0)
    print(predict_id[0])
    print(target[0])
    rid_acc = (predict_id == target).sum() / sum1
    print(rid_acc)

    exit()
    
    # rid_acc = cnt / ttl
    rid_recall = 0 #correct_id_num / ttl_trg_id_num
    rid_precision = 0 #correct_id_num / ttl_pre_id_num
    return rid_acc.item(), rid_recall, rid_precision