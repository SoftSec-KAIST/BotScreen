from utils import *
e = exp_from_arguments()        # get arguments

#eprint(f'### model_dir: {e.model_dir} ###' +'\n')   # TODO: change

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
        eprint('... saved gru model found! ...\n')
        model_loaded = True
    except:
        model_loaded = False

    # train model if the model failed to load or different config
    if not e.chk_config or not model_loaded:
        eprint('... training gru ...\n')

        data_train[e.data_cols] = normalize(df=data_train[e.data_cols],
                                            norm_args=e.norm_args).ewm(alpha=0.9).mean()

        aim_train = AimDataset(data=data_train,
                               data_cols=e.data_cols,
                               stride=e.stride,
                               win_size=e.win_size)

        # define model
        model = eval(e.model)(n_tags=len(e.data_cols),
                              n_hidden=e.n_hidden,
                              n_layer=e.n_layer).cuda()

        # training
        model.train()
        best_model, loss_hist = train(aim_train, model, e.bat_size, e.n_epoch)
        saved_model = {
                'args': (len(e.data_cols), e.n_hidden, e.n_layer),
                'state': best_model['state'],
                'best_epoch': best_model['epoch'],
                'loss_history': loss_hist,
            }

        # save trained model
        eprint('... gru trained, saving ... ')
        torch.save(saved_model, model_dir)
        eprint('done\n')

    eprint('done\n\n')

# renew config
if not e.chk_config:
    config = json.dumps(e.model_cfg, indent=4, sort_keys=True)
    with open('trained_models/config.json', 'w') as f:
        f.write(config)