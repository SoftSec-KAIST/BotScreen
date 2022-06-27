# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

# placeholder for results
list_tot, list_neg, list_fp, list_fpr = [], [], [], []

# split experiments into k folds
for i,test_games in enumerate(e.splits):

    # run on single split
    if e.split >= 0:
        if i != e.split:
            continue

    eprint(f'### split {i} ###' +'\n')

    train_games = sorted(set(e.all_games) - set(test_games))

    # directories for gru models
    model_dir = f'trained_models/gru_k{i}.pt'

    data_train = pd.concat(smooth_df(v) for (g,o,p),v in e.frames.items()\
                            if g in train_games and o==p and p not in e.cheater[g])

    e.norm_args = data_train[e.data_cols].min(), data_train[e.data_cols].max()

    ### configuring model ###
    eprint('configuring models ...\n')

    # try to load pretrained model
    try:
        saved_model = torch.load(model_dir, map_location=f'cuda:{e.gpu}')
        eprint('... saved gru model found! loading model ... ')
        model = eval(e.model)(*saved_model['args']).cuda()
        model.load_state_dict(saved_model['state'])
        model_loaded = True
    except:
        model_loaded = False

    # error if the model failed to load or in different config
    if not e.chk_config or not model_loaded:
        eprint('error!\n')
        exit()

    eprint('done\n')

    # evaluation mode
    model.eval()

    ##### evaluation #####
    eprint('evaluation ...\n')

    ### directory for pre-evaluated results
    eval_dir = f'trained_models/eval_k{i}'

    # try to load eval results
    try:
        eval_trues, eval_scores = pickle.load(open(eval_dir,'rb'))
        assert set(eval_trues.keys()) == set(test_games)
        eval_loaded = True
    except:
        eval_loaded = False

    # error when pre-evaluated data cannot be loaded or config mismatch
    if not e.chk_config or not eval_loaded:
        eprint('error!\n')
        exit()

    # aggregated true labels and boundary error scores
    t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
    s_agg = sum([get_median(s) for s in eval_scores.values()], [])

    # determine optimal threshold
    thresh = get_thresh(s_agg, t_agg, maximize='prec')

    # for each game in test games
    for g in test_games:
        # get number of players from experiment info
        _,l = e.exp_info[g[0]]
        player_id = range(l)

        for o in player_id:
            cnt_tot = 0
            cnt_fp = 0
            cnt_neg = 0
            for p in player_id:
                true_label, score =\
                    anom_score(m=model,
                               e=e,
                               key=(g,str(o),str(p)),
                               use_all=False,
                               return_all=True)

                cnt_tot += len(score)

                # only for benign user
                if not true_label:
                    score = np.array(score)
                    score = score > thresh    # above threshold

                    # get number of negatives and false positives
                    cnt_neg += len(score)
                    cnt_fp += np.sum(score)

            # append results to each list
            list_tot.append(cnt_tot)
            list_neg.append(cnt_neg)
            list_fp.append(cnt_fp)
            if cnt_neg > 0:
                list_fpr.append(cnt_fp/cnt_neg)

# print results
print("Avg. # of reports by each player per game:", np.mean(list_tot))  # average reports
print("Avg. # of FPs by each player per game:", np.mean(list_fp))       # average FPs
print("Avg. FPR by each player per game:", np.mean(list_fpr))           # average FPR