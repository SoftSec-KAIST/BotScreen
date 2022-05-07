# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

# arguments for repeated experiments
seeds = range(5)
n_atks = range(13)
attackers = ['dishonest (flip)',
             'dishonest (random)',
             'dishonest-but-rational']

# placeholder and column names for results
res = []
all_col = ["n_split", "split", "win_size", "duration", "seed",
           "atk_seed", "n_atk", "attacker", "best_prec", "n_player"]

# repeat experiments
for atk_seed, attacker, n_atk in product(seeds, attackers, n_atks):

    # split experiments into k folds
    for i,test_games in enumerate(e.splits):

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

            p_agg.extend(p)

        b_prec = get_stats(p_agg, t_agg)[0]
        n_player = len(p_agg)

        info = [e.n_split, i, e.win_size, e.duration, e.seed, atk_seed, n_atk, attacker]
        stats = [b_prec, n_player]
        res.append([*info, *stats])

        eprint('done\n\n')

# weighted average for each repeated experiment
res = pd.DataFrame(res, columns=all_col)
grp_col = ['n_split','win_size','duration','seed','atk_seed','attacker','n_atk']
data_col = ['best_prec']
wgt_col = 'n_player'

# load and sort bench results
res.drop_duplicates(inplace=True)
res.sort_values(by=grp_col, inplace=True)

# weighted average for bench results
res_wa = []
for i,(k,v) in enumerate(res.groupby(grp_col)):
    avg = np.average(v[data_col].values, weights=v[wgt_col].values, axis=0)
    res_wa.append([*k, *avg])
res_wa = pd.DataFrame(res_wa, columns=grp_col+data_col)

# save results
if e.save_results:
    os.makedirs('bench',exist_ok=True)
    if not os.path.exists('bench/bench_atk.tsv'):
        with open('bench/bench_atk.tsv','w') as f:
            f.write('\t'.join(all_col))
            f.write('\n')

    for _,row in res.iterrows():
        r = row.values
        info = map(str,r[:-2])

        with open('bench/bench_atk.tsv','a') as f:
            f.write('\t'.join(info))
            f.write('\t' + f'{r[-2]:.4f}')
            f.write('\t' + str(int(r[-1])))
            f.write('\n')

    res_wa.to_csv('bench/bench_atk_wa.tsv', sep='\t', index=False, float_format='%.4f')

# plot results
os.makedirs('figures',exist_ok=True)
plt.rcParams.update({'font.size': 9})
sns.lineplot(x='n_atk',
             y='best_prec',
             hue='attacker',
             data=res)
plt.legend(loc='lower left')
plt.xlabel('number of dishonest players')
plt.ylabel('accuracy')
plt.savefig('figures/fig_07_atk.pdf')
