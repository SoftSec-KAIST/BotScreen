##### Dummy code for preprocessing #####

import os, sys
import csv, argparse
import numpy as np
import pandas as pd
from scipy.spatial import distance

# field names
TS_FIELD = 'timestamp'
EVENT_FIELD = [TS_FIELD,'event','src','dst']
PLAYER_FIELD = ['is_bot','loc_x','loc_y','loc_z','s_x','s_y','s_z']
BOT_FIELD = [TS_FIELD,'bot_on','player']
# time interval
INTERVAL = 16

def eprint(s):  sys.stderr.write(s)

def gen_field(i, names):
    return [f'{n}_{i}' for n in names]

def sph_to_car(r, the, phi):
    v = np.array([r*np.cos(phi)*np.sin(the),
                  r*np.sin(phi)*np.sin(the),
                  r*np.cos(the)])
    return v

def interp(raw_data, raw_ts, new_ts):
    assert len(raw_data) == len(raw_ts)
    return [round(np.interp(i,raw_ts,raw_data),10) for i in new_ts]

def process_row(row, fields, player_id, enemies, offset, w):
    n = len(fields)

    d = {TS_FIELD: str(int(row[0])-offset)}  # apply offset to the timestamp

    n_chunks = int(len(row)/n)
    chunks = (row[n*i+1:n*(i+1)+1] for i in range(n_chunks))

    for c in chunks:
        # skip if pid not in the list
        if c[1] not in player_id:   continue

        c[1] = player_id.index(c[1])

        c = [float(i) for i in c]

        # preprocess spherical coordinate values
        if c[5] > 90:           c[5] -= 360
        elif c[5] < -90:        c[5] += 360

        c[5] = 90 - c[5]

        if c[6] < 0:            c[6] += 360
        elif c[6] >= 360:       c[6] -= 360

        assert c[5] <= 180 and c[5] >= 0
        assert c[6] >= 0 and c[6] < 360

        keys = gen_field(int(c[1]), fields)       # get keys

        # convert spherical coordinates into cartesian coordinates
        v = sph_to_car(1,float(c[5])*np.pi/180,float(c[6])*np.pi/180)
        values = c[:1] + c[2:5]                 # exclude player id
        values = [*values, *v]

        d.update({k:round(v,10) for k,v in zip(keys,values)})

    d = process_angle(d, fields, enemies)
    w.writerow(d)


def process_angle(d, fields, enemies):
    players = set([k.split('_')[-1] for k in d.keys()]) - {TS_FIELD}

    locs = {p: [d[f'{s}_{p}'] for s in ['loc_x','loc_y','loc_z']] for p in players}
    sights = {p: [d[f'{s}_{p}'] for s in ['s_x','s_y','s_z']] for p in players}

    for obs in players:
        dist_ang, dist_perp = np.inf, np.inf
        others = players - {obs}
        l_obs = np.array(locs[obs])
        s_obs = np.array(sights[obs])
        for p in others.intersection(enemies):
            l_tar = np.array(locs[p])
            d_tar = l_tar - l_obs

            d_ang = distance.cosine(d_tar, s_obs)

            if dist_ang > d_ang and d_ang < 0.15:
                dist_ang = d_ang
                dist_perp = np.linalg.norm(np.cross(d_tar, s_obs)) / np.linalg.norm(s_obs)
                dist_euc = np.linalg.norm(d_tar)
                target = p

        if dist_ang != np.inf and dist_perp != np.inf:
            d[f'd_a_{obs}'] = dist_ang
            d[f'd_p_{obs}'] = dist_perp
            d[f'd_e_{obs}'] = dist_euc
            d[f'tar_{obs}'] = target

    return d


def merge_all(raw_dir, process_dir, player_id, enemies, offset=0):
    if os.path.exists(f'{process_dir}/log_player_all.csv'):     return

    num_player = len(player_id)
    fields_ext = PLAYER_FIELD + ['d_a','d_p','d_e','tar']
    field_all = [TS_FIELD]
    field_all += sum([gen_field(i,fields_ext) for i in range(num_player)],[])

    with open(f'{raw_dir}/log_player.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        w = open(f'{process_dir}/log_player_all.csv','w')
        writer = csv.DictWriter(w, fieldnames=field_all)
        writer.writeheader()

        for r in reader:
            process_row(r[:-1], PLAYER_FIELD, player_id, enemies, offset, writer)

        w.close()

    # remove duplicates
    log = pd.read_csv(f'{process_dir}/log_player_all.csv')
    log = log.drop_duplicates()
    log.to_csv(f'{process_dir}/log_player_all.csv', index=False)


def interpolate_all(raw_dir, process_dir, player_id, offset, t_base, sub_event=None):
    assert os.path.exists(f'{process_dir}/log_player_all.csv')

    chk_done = [os.path.exists(f'{process_dir}/log_player_{i}.csv') for i,_ in enumerate(player_id)]
    chk_done += [os.path.exists(f'{process_dir}/log_event_processed.csv')]
    chk_done += [os.path.exists(f'{process_dir}/log_bot_processed.csv')]

    # return if everything is already done
    if not False in chk_done:       return

    # load csv files
    df_player = pd.read_csv(f'{process_dir}/log_player_all.csv')
    df_event = pd.read_csv(f'{raw_dir}/log_event.csv', names=EVENT_FIELD)
    df_bot = pd.read_csv(f'{raw_dir}/log_bot_on.csv', names=BOT_FIELD)

    df_event = sub_event if sub_event is not None else df_event

    # normalize and apply offsets to timestamps
    ts_b = df_bot.loc[:,TS_FIELD] - offset
    ts_e = df_event.loc[:,TS_FIELD] - offset
    ts_p = df_player.loc[:,TS_FIELD]            # offsets already applied

    ts_b = ((ts_b - t_base)/INTERVAL).astype('int')
    ts_e = ((ts_e - t_base)/INTERVAL).astype('int')
    ts_p = ((ts_p - t_base)/INTERVAL).astype('int')

    # process log_event
    eprint('\n... processing log_event.csv ... ')
    df_event[TS_FIELD] = ts_e
    df_event = df_event[df_event[TS_FIELD] >= 0].drop_duplicates()
    df_event = df_event[df_event['dst'] != '-1']        # remove fall damage

    # helper function to transform event log
    def indexer(x):
        if str(x) not in player_id:
            return np.nan
        return player_id.index(str(x))

    df_event['src'] = df_event['src'].transform(indexer)
    df_event['dst'] = df_event['dst'].transform(indexer)

    df_event.to_csv(f'{process_dir}/log_event_processed.csv', index=False)
    eprint('done')

    # process log_bot_on
    eprint('\n... processing log_bot_on.csv ... ')
    ts = list(range(min(ts_b),max(ts_b)+1))

    d = df_bot.loc[:,'bot_on']
    bot_on = interp(d, ts_b, ts)

    pid = df_bot['player'][0]
    assert (df_bot['player'] == pid).all() and str(pid) in player_id
    player = [player_id.index(str(pid))]*len(ts)

    df_bot = pd.DataFrame({TS_FIELD: ts, 'bot_on': bot_on, 'player': player})
    df_bot = df_bot[df_bot[TS_FIELD] >= 0].drop_duplicates()
    df_bot.to_csv(f'{process_dir}/log_bot_processed.csv', index=False, columns=BOT_FIELD)
    eprint('done')

    player_field_ = PLAYER_FIELD + ['d_a','d_p','d_e','tar']

    # process log_player
    for i,_ in enumerate(player_id):
        if df_player[f'is_bot_{i}'].count() == 0:     continue

        eprint('\n... processing ')
        eprint(f'log_player_{i}.csv ... ')

        # new dataframe
        df = pd.DataFrame()
        ts = list(range(max(ts_p)+1))
        df[TS_FIELD] = ts

        # interpolation
        for c in gen_field(i,player_field_):
            d = df_player.loc[:,c]
            df[c] = interp(d, ts_p, ts)

        df.columns = [TS_FIELD] + player_field_
        df = df.dropna(subset=PLAYER_FIELD)
        df.to_csv(f'{process_dir}/log_player_{i}.csv', index=False)

        eprint('done')


##### beginning of main #####
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, help='experiment name')
    parser.add_argument('--game', type=int, help='game ID')
    a = parser.parse_args()

    # make sure the data exists
    assert os.path.exists(f'data_raw/{a.exp_name}/')

    eprint(f'exp name: {a.exp_name}, game: {a.game}' + '\n\n')

    # player and cheater names
    player_id = sorted(os.listdir(f'data_raw/{a.exp_name}'))
    # recover player ids with question marks
    player_id_ = [p.replace("_","?") for p in player_id]
    cheater = []

    # get the best recorded log_event
    l_event = 0
    for p in player_id:
        assert os.path.exists(f'data_raw/{a.exp_name}/{p}/{a.game}')

        csv = f'data_raw/{a.exp_name}/{p}/{a.game}/log_event.csv'
        df_event = pd.read_csv(csv, names=EVENT_FIELD)

        if l_event <= len(df_event) and len(df_event) == len(set(df_event[TS_FIELD])):
            l_event = len(df_event)
            best_event = df_event

    # determine teams
    enemy = {}
    for p in player_id:
        csv = f'data_raw/{a.exp_name}/{p}/{a.game}/log_player.csv'

        with open(csv, 'r') as f:
            data = f.read()
            occ = sorted([(data.count(o),(i,o)) for i,o in enumerate(player_id_)])
            enemy[p] = frozenset(v[1] for v in occ[:int(len(player_id)/2)])

    # players are split into two
    assert len(set(enemy.values())) == 2

    teams = sorted(set(enemy.values()))

    # set base time for all observers
    t_base = best_event[TS_FIELD][0]

    # process for each observer
    for id_o, obs in enumerate(player_id):

        # directories for loading and saving data
        raw_dir = f'data_raw/{a.exp_name}/{obs}/{a.game}'
        process_dir = f'data_processed/{a.exp_name}/game_{a.game}/obs_{id_o}'

        assert os.path.exists(f'{raw_dir}/log_event.csv')

        # make directories
        os.makedirs(process_dir, exist_ok=True)

        # start processing
        eprint(f'observer {id_o} ...')

        # record cheaters
        bot_on = pd.read_csv(f'{raw_dir}/log_bot_on.csv')
        hack = bot_on.iloc[:,1].values
        assert bot_on.iloc[:,2].values.all()

        # only flag aimhack if aimhack used more than half of the playtime
        if np.mean(hack) > 0.5:
            cheater.append((id_o,obs))

        # calculate time offset based on log_events
        event = pd.read_csv(f'{raw_dir}/log_event.csv', names=EVENT_FIELD)

        if 'hit' not in event['event'].values:
            ref_event = best_event[best_event['event'] == 'fire']
        else:
            ref_event = best_event

        diff_events = len(ref_event) - len(event)
        t_obs = event[TS_FIELD].values                         # time in obs_event
        t_best = ref_event[TS_FIELD][diff_events:].values      # time in ref_event

        # average time diff btw the observed and the best
        offset = np.mean(t_obs - t_best)

        # just use best event when some information is omitted in the event log
        sub_event = None
        if 'hit' not in event['event'].values:
            sub_event = best_event.copy()
            sub_event[TS_FIELD] += offset

        print(offset)
        continue

        # set enemies
        enemies = [str(i) for i,_ in enemy[obs]]

        # merge data
        eprint('\n... cleaning and merging player logs ... ')
        merge_all(raw_dir, process_dir, player_id_, enemies, offset)
        eprint('done\n')

        # process data
        eprint('... processing logs ... ')
        interpolate_all(raw_dir, process_dir, player_id_, offset, t_base, sub_event)
        eprint('\n... done\n\n')

    # record player and cheater ids
    game_dir = f'data_processed/{a.exp_name}/game_{a.game}'
    with open(f'{game_dir}/player_id', 'w') as f:
        #f.write('\n'.join([f'{i}: {p}' for i,p in enumerate(player_id_)]))
        # anonymize
        f.write('\n'.join([f'{i}: player_{i}' for i,p in enumerate(player_id_)]))

    with open(f'{game_dir}/cheater', 'w') as f:
        #f.write('\n'.join([f'{i}: {p.replace("_","?")}' for i,p in sorted(cheater)]))
        # anonymize
        f.write('\n'.join([f'{i}: player_{i}' for i,p in sorted(cheater)]))

    with open(f'{game_dir}/teams', 'w') as f:
        #f.write('\n'.join([str([f'{i}: {p}' for i,p in sorted(k)]) for k in set(enemy.values())]))
        # anonymize
        f.write('\n'.join([str([f'{i}: player_{i}' for i,p in sorted(k)]) for k in set(enemy.values())]))


# run main
if __name__ == '__main__':
    main()