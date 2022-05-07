from utils.util import *
from utils.ml_util import *
from utils.plot_util import *

import argparse

def exp_from_arguments():
    ##### initialization #####
    eprint('initializing ... ')

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--n_layer", type=int, default=3,
                        help="number of hidden units")
    parser.add_argument("--bat_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("--n_epoch", type=int, default=64,
                        help="num of training epochs")
    parser.add_argument("-w", "--win_size", type=int, default=20,
                        help="size of sliding windows")
    parser.add_argument("--stride", type=int, default=2,
                        help="stride")
    parser.add_argument("-d", "--duration", type=int, default=10,       # TODO: duration
                        help="duration of observation")

    parser.add_argument("--seed", type=int, default=3,
                        help="random seed")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu to use")

    parser.add_argument("--n_split", type=int, default=7,
                        help="number of splits used for cross validation")
    parser.add_argument("--split", type=int, default=-1,
                        help="split to test on")

    parser.add_argument("--save_results", type=bool, default=False,
                        help="save results to tsv files")

    arg = parser.parse_args()

    # set seed and gpu
    seed_everything(arg.seed)
    set_gpu(arg.gpu)

    # set model
    arg.model = 'sgru'

    # set model name    TODO: change this?
    #info = [arg.model]
    #info += [k.replace('n_','')[0]+str(v) for k,v in vars(arg).items()][:-6]
    #arg.model_name = '_'.join(info)

    # number of games and players for each experiments
    # exp_name: (n_games, n_players)
    arg.exp_info = {
        '1': (7,8),
        '2': (10,10),
        '3': (6,8),
        '4': (5,10)
    }

    # make directory to save trained model
    os.makedirs('trained_models', exist_ok=True)

    # check if current config matches existing config
    cfg_keys = ['seed', 'n_split', 'n_hidden', 'n_layer', 'bat_size', 'n_epoch', 'win_size', 'stride', 'duration']
    arg.model_cfg = {k: v for k,v in vars(arg).items() if k in cfg_keys}
    try:
        config = json.load(open('trained_models/config.json'))
        arg.chk_config = arg.model_cfg == config
        arg.chk_frames = arg.model_cfg['win_size'] == config['win_size'] and \
                         arg.model_cfg['duration'] == config['duration']
    except:
        arg.chk_config = False
        arg.chk_frames = False

    # data selection
    arg.data_cols = ['s_x', 's_y', 's_z']

    # include distances
    arg.all_cols = ['s_x', 's_y', 's_z', 'd_a', 'd_p', 'd_e']

    # try to load prefiltered frames
    try:
        arg.frames, arg.cheater, arg.times =\
            pickle.load(open('trained_models/_frames','rb'))
        frames_loaded = True
    except:
        frames_loaded = False

    # filter frames if the configuration does not match
    if not arg.chk_frames or not frames_loaded:
        arg.frames, arg.cheater, arg.times = {}, {}, {}
        for e in arg.exp_info.keys():
            exp_prefix = f'data_processed/exp_{e}'
            # process relevant frames and cheater lists
            f, c, t = get_frames(exp_prefix,
                                 arg.all_cols,
                                 arg.data_cols,
                                 arg.win_size,
                                 arg.duration)
            arg.frames.update(f)
            arg.cheater.update(c)
            arg.times.update(t)

        pickle.dump((arg.frames, arg.cheater, arg.times),
                    open('trained_models/_frames','wb'))

    # randomly shuffle splits
    arg.all_games = sorted({g for g,_,_ in arg.frames.keys()})
    random.shuffle(arg.all_games)
    arg.splits = np.array_split(arg.all_games, arg.n_split)

    eprint('done\n')

    return arg


### return max anomaly score
def anom_score(m, e, key, use_all=True):
    """ Get max anomaly scores for given events

    Parameters
    ----------
    m: model
    e: experiment arguments
    key: key
    use_all: use every available data

    Returns
    ----------
    y_true: true label
    mx_score: max score
    """

    d = e.frames[key]

    data = smooth_df(d)
    data[e.data_cols] = normalize(df=data[e.data_cols],
                                  norm_args=e.norm_args).ewm(alpha=0.9).mean()

    aim_ds = AimDataset(data=data,
                        data_cols=e.data_cols,
                        win_size=e.win_size,
                        atk_col='aimhack')

    ts, diff, att = inference(aim_ds, m, e.bat_size)
    score = np.linalg.norm(diff, ord=1.0, axis=1)/diff.shape[1]

    # assert all labels are equal
    atk_labels = put_labels(np.array(att), threshold=0.5)
    assert np.all(atk_labels == atk_labels[0])
    y_true = atk_labels[0]

    # extract timestamp s.t. scores are maximized
    ts_new, diff_new, score_new = process_dist(ts, diff)

    # choose data to use based on the predicate
    score = score if use_all else score_new

    return y_true, max(score)


### return max anomaly score
def anom_times(m, e, key, use_all=True):
    """ Get anomaly scores for the given model

    Parameters
    ----------
    m: model
    e: experiment arguments
    key: key
    use_all: use every available data

    Returns
    ----------
    y_true: true label
    mx_score: max score
    """

    data = smooth_df(e.frames[key])
    data[e.data_cols] = normalize(df=data[e.data_cols],
                                  norm_args=e.norm_args).ewm(alpha=0.9).mean()

    aim_ds = AimDataset(data=data,
                        data_cols=e.data_cols,
                        win_size=e.win_size,
                        atk_col='aimhack')

    ts, diff, att = inference(aim_ds, m, e.bat_size)
    score = np.linalg.norm(diff, ord=1.0, axis=1)/diff.shape[1]

    # assert all labels are equal
    atk_labels = put_labels(np.array(att), threshold=0.5)
    assert np.all(atk_labels == atk_labels[0])
    y_true = atk_labels[0]

    # extract timestamp s.t. scores are maximized
    ts_new, diff_new, score_new = process_dist(ts, diff)

    # choose data to use based on the predicate
    score = score if use_all else score_new

    return score, ts_new


### return max anomaly stats
def anom_stat(e, key, use_all=True, stat='acca'):
    """ Get anomaly scores for the given model

    Parameters
    ----------
    m: model
    e: experiment arguments
    key: key
    use_all: use every available data

    Returns
    ----------
    y_true: true label
    mx_score: max score
    """

    data = smooth_df(e.frames[key])

    # calculate delta
    vara = data[e.data_cols].diff().apply(np.linalg.norm, axis=1)
    acca = vara.diff().apply(abs)
    ts = acca.index.values

    scores = eval(stat)

    new_ts, new_score = [], []
    for g in consecutive_groups(zip(ts,scores), lambda x: x[0]):
        group = list(g)

        try:
            idx = np.nanargmax([s for _,s in group])
            new_ts.append(group[idx][0])
            new_score.append(group[idx][1])
        except:
            continue

    # assert all labels are equal
    y_true = data['aimhack'].all()

    return y_true, np.nanmax(new_score)