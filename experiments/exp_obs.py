# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

# placeholders for results
res = {'accuracy': [],
       'measure': [],
       'team': []}
accs, obsrates = [], []

# for each split
for i,test_games in enumerate(e.splits):

    # run on single split
    if e.split >= 0:
        if i != e.split:
            continue

    eprint(f'### split {i} ###' +'\n')

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

    loss_list = []
    team_list = []

    # load loss information for each test game
    for g in test_games:
        exp, game = g.split('_')
        exp_name = f'exp_{exp}'
        loss_dir = f'data_loss/{exp_name}/game_{game}'
        try:
            loss_obs, _, _ = pickle.load(open(loss_dir, 'rb'))
        except:
            eprint('error!\n')
            exit()

        loss_list.extend(loss_obs.T.flatten())

        teams = get_teams(exp_name, game)
        same_team = team_indicator(teams)
        same_team[np.eye(len(same_team), dtype=bool)] = np.nan

        team_list.extend(same_team.flatten())

    # placeholder for individual predictions
    p_list = []

    t_list = np.concatenate([t.flatten() for t in eval_trues.values()])
    s_list = np.concatenate([s.flatten() for s in eval_scores.values()])

    # determine threshold based on scores and labels
    thresh = get_thresh(s_list, t_list, maximize='acc')
    p_list = np.concatenate([(s > thresh).flatten() for s in eval_scores.values()])

    eprint('done\n\n')

    loss_list = np.array(loss_list)
    team_list = np.array(team_list)

    correct = t_list == p_list

    # observation rates for allies and enemies
    obs_same = sum(loss_list[team_list == True])/sum(team_list == True)
    obs_diff = sum(loss_list[team_list == False])/sum(team_list == False)

    # prediction accuracies for allies and enemies
    acc_same = sum(correct[team_list == True])/sum(team_list == True)
    acc_diff = sum(correct[team_list == False])/sum(team_list == False)

    # record results
    res['accuracy'].extend([acc_same, acc_diff, obs_same, obs_diff])
    res['measure'].extend(['accuracy', 'accuracy', 'obsrate', 'obsrate'])
    res['team'].extend(['ally', 'enemy', 'ally', 'enemy'])

    accs.extend([acc_same, acc_diff])
    obsrates.extend([obs_same, obs_diff])

# plot results
res = pd.DataFrame(res)

os.makedirs('figures',exist_ok=True)
sns.set_theme(style="whitegrid")
sns.boxplot(x='team', y='accuracy', hue='measure', data=res)
plt.legend()
plt.savefig('figures/fig_06_obsrate.pdf')

# calculate Pearson Correlation Coefficient
pcc = np.corrcoef(obsrates, accs)[0,1]
print(f'Pearson Correlation Coefficient: {pcc}')