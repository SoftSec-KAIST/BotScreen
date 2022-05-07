from utils import *
e = exp_from_arguments()        # get arguments

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

    # try to load existing eval results
    try:
        eval_trues, eval_scores = pickle.load(open(eval_dir,'rb'))
        assert set(eval_trues.keys()) == set(test_games)

    except:
        # true labels and error scores for each game
        eval_trues, eval_scores = {}, {}

        # for each game in test games
        for g in test_games:
            # get number of players from experiment info
            _,l = e.exp_info[g[0]]
            player_id = range(l)

            trues, scores = np.zeros((l,l)), np.zeros((l,l))
            for o,p in product(player_id, repeat=2):
                trues[o,p], scores[o,p] = \
                                anom_score(m=model,
                                        e=e,
                                        key=(g,str(o),str(p)),
                                        use_all=False)

            eval_trues[g] = trues
            eval_scores[g] = scores

        # save results
        pickle.dump((eval_trues, eval_scores), open(eval_dir,'wb'))

    eprint('done\n\n')