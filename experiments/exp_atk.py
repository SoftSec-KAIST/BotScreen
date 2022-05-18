# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

# arguments for repeated experiments
n_atks = range(11)
seeds = range(5)
attackers = ['dishonest (flip)',
             'dishonest (random)',
             'dishonest-but-rational (all)',
             'dishonest-but-rational (allies)']

# placeholder and column names for results
res = []
all_col = ["n_split", "split", "win_size", "duration", "seed",
           "atk_seed", "attacker", "q_atk", "best_prec", "n_player"]

# repeat experiments
for attacker, atk_seed, n_atk in product(attackers, seeds, n_atks):

    # split experiments into k folds
    for i,test_games in enumerate(e.splits):

        info = [e.n_split, i, e.win_size, e.duration, e.seed, atk_seed, attacker]

        eprint(f'### split {i} ###' +'\n')

        ##### evaluation #####
        eprint('evaluation ...\n')

        ### directory for pre-evaluated results
        eval_dir = f'trained_models/eval_k{i}'

        # try to load eval results
        try:
            eval_trues, eval_scores = pickle.load(open(eval_dir,'rb'))
            assert eval_trues.keys() == eval_scores.keys()
            eval_loaded = True
        except:
            eval_loaded = False

        # error when pre-evaluated data cannot be loaded or config mismatch
        if not e.chk_config or not eval_loaded:
            eprint('error!\n')
            exit()

        same_teams = {}
        for g in test_games:
            exp, game = g.split('_')
            exp_name = f'exp_{exp}'
            teams = get_teams(exp_name, game)
            same_teams[g] = team_indicator(teams)

        # placeholder for aggregated predictions
        p_agg = []

        t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
        s_agg = sum([get_median(s) for s in eval_scores.values()], [])

        # determine threshold based on scores and labels
        thresh = get_thresh(s_agg, t_agg, maximize='prec')

        for g,scores in eval_scores.items():
            preds = scores > thresh

            # prediction under attack
            p = get_votes_attack(preds,
                                 teams=same_teams[g],
                                 n_atk=n_atk,
                                 attacker=attacker,
                                 seed=atk_seed)
            t = get_votes(eval_trues[g])[0]

            # percentage and accuracy
            q_atk = min(n_atk/len(p), 1.0)
            acc_atk = get_stats(p, t)[0]
            stats = map(lambda s: f'{s:.4f}', [q_atk, acc_atk])
            res.append([*info, *stats, len(p)])

        eprint('done\n\n')

# weighted average for each repeated experiment
res = pd.DataFrame(res, columns=all_col)

# load and sort bench results
res.drop_duplicates(inplace=True)

# save results
if e.save_results:
    os.makedirs('bench',exist_ok=True)
    res.to_csv('bench/bench_atk.tsv', sep='\t', index=False, float_format='%.4f')

# filter by number of participants
N = 10
res = res[res['n_player']==N]
res['q_atk'] = res['q_atk'].astype(float) * N
res['best_prec'] = res['best_prec'].astype(float)

# plot results
sns.set_theme(style='whitegrid', palette=None)
sns.set_context('paper')
sns.lineplot(x='q_atk',
             y='best_prec',
             hue='attacker',
             style='attacker',
             markers=True,
             dashes=False,
             data=res)

plt.legend(loc='lower left')
plt.xlabel('Number of dishonest players')
plt.xlim((0,N))
plt.ylim((0.0,1.0))
plt.ylabel('Aimbot prediction accuracy')
plt.savefig('figures/fig_07_atk.pdf')
