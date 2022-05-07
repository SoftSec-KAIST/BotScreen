from functools import reduce
from itertools import product
import pandas as pd
import numpy as np
import os, pickle

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

def loss_info(p, o, game_dir):
    try:
        csv_true = pd.read_csv(f'{game_dir}/obs_{p}/log_player_{p}.csv')
        ev_true = pd.read_csv(f'{game_dir}/obs_{p}/log_event_processed.csv')
    except:
        return np.nan, np.nan, np.nan

    if p == o:
        return 1.0, 0.0, 0.0

    try:
        csv_obs = pd.read_csv(f'{game_dir}/obs_{o}/log_player_{p}.csv')
        ev_obs = pd.read_csv(f'{game_dir}/obs_{o}/log_event_processed.csv')
    except:
        return 0.0, np.nan, np.nan

    fire_true = ev_true['event'] == 'fire'
    src_true = ev_true['src'] == p
    rel_true = ev_true[fire_true & src_true]['timestamp']

    ts_rel_true = map(lambda t: range(t-60,t+60), rel_true)
    ts_sorted_true = reduce(lambda x,y: set(x).union(y), ts_rel_true)

    fire_obs = ev_obs['event'] == 'fire'
    src_obs = ev_obs['src'] == p
    rel_obs = ev_obs[fire_obs & src_obs]['timestamp']

    ts_rel_obs = map(lambda t: range(t-60,t+60), rel_obs)
    ts_sorted_obs = reduce(lambda x,y: set(x).union(y), ts_rel_obs)

    ts_true_rel = set(csv_true['timestamp']).intersection(ts_sorted_true)
    ts_obs_rel = set(csv_obs['timestamp']).intersection(ts_sorted_obs)
    ts_inter = ts_true_rel.intersection(ts_obs_rel)

    if len(ts_inter) == 0:
        return 0.0, np.nan, np.nan

    csv_true = csv_true[csv_true['timestamp'].isin(ts_inter)].set_index('timestamp')
    csv_obs = csv_obs[csv_obs['timestamp'].isin(ts_inter)].set_index('timestamp')

    diff = csv_true - csv_obs
    col_aim = ['s_x', 's_y', 's_z']
    col_loc = ['loc_x', 'loc_y', 'loc_z']

    l_aim = np.linalg.norm(diff[col_aim], axis=1)
    mse_aim = np.mean(l_aim) * 0.5
    std_aim = np.std(l_aim)

    l_loc = np.linalg.norm(diff[col_loc], axis=1)
    mse_loc = np.mean(l_loc)
    std_loc = np.std(l_loc)

    return len(ts_inter)/len(ts_true_rel), mse_aim, mse_loc


def plot_heatmap(data, cmap=plt.cm.Blues, maxval=None, title=None, filename=None):
    l = len(data)
    fig, ax = plt.subplots(figsize=(6,5), dpi=400)
    im = ax.pcolor(data, cmap=cmap, vmin=0, vmax=maxval)
    fig.colorbar(im)

    if title is not None:
        ax.set_title(title, y=1.1)

    ax.set_xlabel('observer_id')
    ax.set_ylabel('player_id')

    ax.tick_params(direction="in")
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    for (j,i),d in np.ndenumerate(data):
        ax.text(i+0.5,j+0.5,f'{d:.2f}',ha='center',va='center')

    tick_pos = np.arange(0.5,l,1.0)
    ax.set_xticks(tick_pos, minor=False)
    ax.set_yticks(tick_pos, minor=False)
    ax.set_xticklabels(range(l), minor=False)
    ax.set_yticklabels(range(l), minor=False)
    plt.tight_layout()
    plt.gca().set_aspect('equal')

    if filename is not None:
        fig.savefig(filename)

    plt.close()


def loss_matrix(game, exp_name, plt_loss=False):
    # read player information
    game_dir = f'data_processed/{exp_name}/game_{game}'
    pid = f'{game_dir}/player_id'
    try:
        with open(pid, 'r') as f:
            l = len([line.strip('\n') for line in f])
        players = range(l)
    except:
        print('No game information!')
        return

    # directory names
    os.makedirs(f'data_loss/{exp_name}', exist_ok=True)
    dataname = f'data_loss/{exp_name}/game_{game}'
    try:
        loss_obs, loss_aim, loss_loc = pickle.load(open(dataname, 'rb'))
    except:
        loss_obs, loss_aim, loss_loc = np.zeros((l,l)), np.zeros((l,l)), np.zeros((l,l))
        for p,o in product(players, repeat=2):
            loss_obs[p,o], loss_aim[p,o], loss_loc[p,o] = loss_info(p, o, game_dir)

        pickle.dump((loss_obs, loss_aim, loss_loc), open(dataname, 'wb'))

    # plot losses
    if plt_loss:
        os.makedirs(f'data_loss/{exp_name}/figures', exist_ok=True)
        plot_heatmap(data=loss_obs,
                    cmap=plt.cm.Blues,
                    maxval=1.0,
                    title=f'Observation rate (game {game})',
                    filename=f'data_loss/{exp_name}/figures/game_{game}_loss_obs.png')

        plot_heatmap(data=loss_aim,
                    cmap=plt.cm.Greens,
                    maxval=1.0,
                    title=f'Average RMSE of aim (game {game})',
                    filename=f'data_loss/{exp_name}/figures/game_{game}_loss_aim.png')

        plot_heatmap(data=loss_loc,
                    cmap=plt.cm.Greys,
                    title=f'Average RMSE of location (game {game})',
                    filename=f'data_loss/{exp_name}/figures/game_{game}_loss_loc.png')

exp_info = {
    '1': (7,8),
    '2': (10,10),
    '3': (6,8),
    '4': (5,10)
}

# main
def main():
    for exp,v in exp_info.items():
        n_games,_ = v
        exp_name = f'exp_{exp}'

        for g in range(1,n_games+1):
            loss_matrix(g, exp_name, plt_loss=True)

# run main
if __name__ == '__main__':
    main()