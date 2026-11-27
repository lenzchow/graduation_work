import pandas as pd
from .temporal_cond import timestamp_list_to_minutes, compute_waiting_time

LAT_MIN, LAT_MAX = 22.705, 22.890
LON_MIN, LON_MAX = 108.2, 108.46

def filter_bs_in_box(df_bs,
                     lat_col='c_lat', lon_col='c_lon',
                     lat_min=LAT_MIN, lat_max=LAT_MAX,
                     lon_min=LON_MIN, lon_max=LON_MAX):
    m = (
        (df_bs[lat_col] >= lat_min) & (df_bs[lat_col] <= lat_max) &
        (df_bs[lon_col] >= lon_min) & (df_bs[lon_col] <= lon_max)
    )
    return df_bs[m].reset_index(drop=True)

def preprocess_traj(df_traj, df_bs, min_len, max_len):
    valid_uli = set(df_bs['c_uli'].astype(str))
    df_traj = df_traj[df_traj['c_uli'].astype(str).isin(valid_uli)]

    out = []
    for (uid, day), g in df_traj.groupby(['c_number','c_day']):
        if not (min_len <= len(g) <= max_len):
            continue
        g = g.sort_values('c_timestamp')
        traj = g['c_uli'].astype(str).tolist()
        traj_ts = list(zip(traj, g['c_timestamp']))
        tim = timestamp_list_to_minutes(traj_ts)
        sta = compute_waiting_time(tim)
        out.append([uid, day, traj, traj_ts, tim, sta])

    return pd.DataFrame(
        out,
        columns=['c_number','c_day','traj','traj_timestamp','tim','sta']
    )
