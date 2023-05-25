# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *

def cart2sph(x, y, z):
    xy = np.sqrt(x ** 2 + y ** 2)

    x_2 = x**2
    y_2 = y**2
    z_2 = z**2

    r = np.sqrt(x_2 + y_2 + z_2)
    theta = np.arctan2(y, x) * 180 / np.pi
    phi = np.arctan2(xy, z) * 180 / np.pi

    return theta, phi

def velo(x, y):
    xd = x[0] - y[0]
    yd = x[1] - y[1]
    if xd > 180:
        xd = 360 - xd
    elif xd < -180:
        xd = 360 + xd

    if yd > 180:
        yd = 360 - yd
    elif yd < -180:
        yd = 360 + yd
    return np.sqrt(xd ** 2 + yd ** 2)

def gacc(lst):
    vel = []
    acc = []
    lst = list(map(lambda x: cart2sph(*x), lst))
    for i in range(len(lst)-1):
        vel.append(velo(lst[i+1], lst[i]))
    for i in range(len(vel)-1):
        acc.append(np.abs(vel[i+1] - vel[i]))
    return acc

bins = []

for game in sorted(glob('data_processed/*/*')):
    print(game)
    f_cheat = f'{game}/cheater'
    with open(f_cheat, 'r') as f:
        rows = f.readlines()
        cheater = [int(c[0]) for c in rows] if len(rows) else []

    f_player = f'{game}/player_id'
    with open(f_player, 'r') as f:
        rows = f.readlines()
        player_id = [int(c[0]) for c in rows]

    ds = [[] for i in player_id]

    for j,o in enumerate(player_id):
        log_player = f'{game}/obs_{o}/log_player_{o}.csv'
        df = pd.read_csv(log_player)

        now_target = -1
        now_data = []
        for c, d in df.iterrows():
            if np.isnan(d["tar"]):
                if now_target != -1:
                    now_target = -1
                    r = gacc(now_data)
                    ds[o] += r
                    now_data = []
            else:
                target = int(d["tar"])
                if now_target != target and len(now_data) != 0:
                    now_target = target
                    r = gacc(now_data)
                    ds[o] += r
                    now_data = []
                else:
                    now_target = target
                    now_data.append((d["s_x"], d["s_y"], d["s_z"]))

    bins.append((cheater, ds))

#bins = list(filter(lambda x: len(x[0]) == 2, bins))

import random
random.seed(0)
random.shuffle(bins)

from scipy.stats import ks_2samp

# 7 fold
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(7): # test set
    benign_ds = []
    aimbot_ds = []

    for j in range(len(bins)):
        if (j // 4) != i:
            cheater, dat = bins[j]
            for k in range(len(dat)):
                if k not in cheater:
                    benign_ds += dat[k]
                else:
                    aimbot_ds += dat[k]
    for j in range(4):
        cheater, dat = bins[4 * i + j]
        for k in range(len(dat)):
            for tt in range(1):
                test_samp = random.sample(dat[k], 100)
                gult = 0
                ngult = 0
                for tran_c in range(300):
                    r = random.randrange(2)
                    if r:
                        tran_samp = random.sample(benign_ds, 100)
                    else:
                        tran_samp = random.sample(aimbot_ds, 100)
                    if ks_2samp(tran_samp, test_samp).pvalue < 0.05:
                        if r:
                            gult += 1
                        else:
                            ngult += 1
                    else:
                        if r:
                            ngult += 1
                        else:
                            gult += 1
                if gult >= ngult:
                    if k in cheater:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if k in cheater:
                        FN += 1
                    else:
                        TN += 1

# performance measures
acc = float(TP + TN)/float(TP + TN + FN + FP)
fpr = float(FP)/float(FP + TN)
fnr = float(FN)/float(FN + TP)

print('Existing method: ks_AccA')
print('accuracy: {:.4f} ({}/{})'.format(acc, TP+TN, TP+TN+FP+FN))
print('fpr: {:.4f} ({}/{})'.format(fpr, FP, FP+TN))
print('fnr: {:.4f} ({}/{})'.format(fnr, FN, FN+TP))
print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')