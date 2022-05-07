# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

stdevs = []
agree = []
# split experiments into k folds
for i,test_games in enumerate(e.splits):

    # run on single split
    if e.split >= 0:
        if i != e.split:
            continue

    eprint(f'### split {i} ###' +'\n')

    train_games = sorted(set(e.all_games) - set(test_games))

    # directories for gru and svm models
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
        eprint('error! no pre-trained model\n')
        exit()

    eprint('done\n')

    # evaluation mode
    model.eval()

    ##### evaluation #####
    eprint('evaluation ...\n')

    ### directory for pre-evaluated results
    eval_dir = f'trained_models/eval_k{i}'

    try:
        eval_trues, eval_scores = pickle.load(open(eval_dir,'rb'))
        assert set(eval_trues.keys()) == set(test_games)

    except:
        eprint('error! no pre-evaluated data\n')
        exit()

    # placeholder for aggregated predictions
    p_agg = []

    t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
    s_agg = sum([get_median(s) for s in eval_scores.values()], [])

    # determine threshold based on scores and labels
    thresh = get_thresh(s_agg, t_agg, maximize='prec')

    # for each game in test set
    for g in test_games:
        # get number of players from experiment info
        _,l = e.exp_info[g[0]]
        player_id = range(l)

        scores, times = {}, {}
        for o,p in product(player_id, repeat=2):
            scores[(o,p)], times[(o,p)] = anom_times(m=model,
                                                     e=e,
                                                     key=(g,str(o),str(p)),
                                                     use_all=False)

        for p in player_id:
            for s,t in zip(scores[(p,p)],times[(p,p)]):
                ss = [s]
                for o in player_id:
                    idx, dst = closest(t, times[(o,p)])
                    if dst > 60:
                        continue
                    else:
                        ss.append(scores[(o,p)][idx])

                stdevs.append(np.std(ss))
                res = np.array(ss) < thresh
                l_t, l_f = len(res[res==True]), len(res[res==False])
                agree.append(max(l_t,l_f)/len(res))

# plot results
os.makedirs('figures',exist_ok=True)
sns.histplot(data=stdevs, stat='percent', bins=50)
plt.xlabel('Standard deviation of anomaly scores')
plt.savefig('figures/fig_05_std.pdf')
print(f'Agreement: {np.mean(agree)}')