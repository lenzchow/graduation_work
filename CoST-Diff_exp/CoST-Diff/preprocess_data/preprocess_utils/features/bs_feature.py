import numpy as np
import pandas as pd
import json, os

def lac_feature(df_bs):
    onehot_lac = pd.get_dummies(df_bs['c_lac'].astype(str)).to_numpy()
    lat = np.deg2rad(df_bs['c_lat'].to_numpy())
    lon = np.deg2rad(df_bs['c_lon'].to_numpy())
    geo = np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)], 1)
    ac = pd.get_dummies(df_bs['c_areacode'].astype(str)).to_numpy()
    return np.hstack([onehot_lac, geo, ac]).astype(np.float32)

def save_bs_feat(df_bs, uli_index_dict, out_dir):
    feat = lac_feature(df_bs)
    np.save(os.path.join(out_dir,'bs_feat.npy'), feat)
