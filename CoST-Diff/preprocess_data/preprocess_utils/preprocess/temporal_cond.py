# weekday / slot

# temporal_cond.py
from datetime import datetime

def weekday_from_cday(c_day: int) -> int:
    return datetime.strptime(str(c_day), "%Y%m%d").weekday()

def timestamp_list_to_minutes(traj_ts):
    return [
        round(datetime.fromtimestamp(ts).hour * 60 +
              datetime.fromtimestamp(ts).minute +
              datetime.fromtimestamp(ts).second / 60, 2)
        for _, ts in traj_ts
    ]

def waiting_time(tim):
    return [round(tim[i+1]-tim[i], 2) for i in range(len(tim)-1)]

def build_slot_seq(tim, is_weekday):
    if not is_weekday:
        return [0]*len(tim)
    slots = []
    for m in tim:
        if 420 <= m < 540: slots.append(1)
        elif 1020 <= m < 1200: slots.append(2)
        else: slots.append(0)
    return slots
