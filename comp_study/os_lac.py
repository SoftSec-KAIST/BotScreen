# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  a = dotproduct(v1, v2) / (length(v1) * length(v2))
  if a >= 1.0:
    return 0.0
  return math.acos(a)

TP = 0
TN = 0
FP = 0
FN = 0

for game in sorted(glob('data_processed/*/*')):
    print(f'processing {game} ...')
    f_cheat = f'{game}/cheater'
    with open(f_cheat, 'r') as f:
        rows = f.readlines()
        cheater = [int(c[0]) for c in rows] if len(rows) else []

    f_player = f'{game}/player_id'
    with open(f_player, 'r') as f:
        rows = f.readlines()
        player_id = [int(c[0]) for c in rows]

    pd_lst = {}
    for pid in player_id:
        pid_player = f'{game}/obs_{pid}/log_player_{pid}.csv'
        pd_read = pd.read_csv(pid_player)
        pd_lst[pid] = pd_read

    detect_cheater = set()

    log_event = f'{game}/obs_0/log_event_processed.csv'
    log_event = pd.read_csv(log_event)
    kill = log_event['event'] == 'dead'
    kill_event = log_event[kill]
    for c, d in kill_event.iterrows():
        time, _, src, dst = d
        src_n = src
        if math.isnan(dst):
            continue
        dst = int(dst)
        filter_src = (pd_lst[src]['timestamp'] <= time) & (pd_lst[src]['timestamp'] > time - (500 / 16 + 1))
        filter_src = pd_lst[src][filter_src]
        filter_dst = (pd_lst[dst]['timestamp'] <= time) & (pd_lst[dst]['timestamp'] > time - (500 / 16 + 1))
        filter_dst = pd_lst[dst][filter_dst]

        agl = 0
        vel = 0
        bef_agl = None
        bef_aim = None
        for sr, ds in zip(filter_src.iterrows(), filter_dst.iterrows()):
            sr = sr[1]
            ds = ds[1]
            tar_vec = (ds["loc_x"] - sr["loc_x"], ds["loc_y"] - sr["loc_y"], ds["loc_z"] - sr["loc_z"])
            aim_vec = (sr["s_x"], sr["s_y"], sr["s_z"])

            if bef_agl is not None:
                agl = angle(tar_vec, aim_vec) / math.pi * 180
                vel = angle(bef_aim, aim_vec) / math.pi * 180

            if bef_agl is not None and agl < bef_agl * 0.2 and vel > 10.0:
                detect_cheater.add(src_n)
                break

            bef_agl = agl
            bef_aim = aim_vec

    for p in player_id:
        if p in cheater and p in detect_cheater:
            TP += 1
        elif p not in cheater and p not in detect_cheater:
            TN += 1
        elif p in cheater and p not in detect_cheater:
            FN += 1
        elif p not in cheater and p in detect_cheater:
            FP += 1

# performance measures
acc = float(TP + TN)/float(TP + TN + FN + FP)
prec = float(TP)/float(TP + FP)
rec = float(TP)/float(TP + FN)

print('\nOpen-source: Little-Anti-Cheat')
print(f'accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}')
print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
