# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

##### evaluation #####
eprint('evaluation ...\n')

eval_trues, eval_scores = {}, {}

# for each game in test set
for g in e.all_games:
    # get number of players from experiment info
    _,l = e.exp_info[g[0]]
    player_id = range(l)

    trues, scores = np.zeros((l,l)), np.zeros((l,l))
    for o,p in product(player_id, repeat=2):
        trues[o,p], scores[o,p] = \
                        anom_stat(e=e,
                                  key=(g,str(o),str(p)),
                                  use_all=False,
                                  stat='acca')

    eval_trues[g] = trues
    eval_scores[g] = scores

# evaluate prediction performances
b_acc, b_prec, auc, _ = eval_preds(eval_trues, eval_scores)

eprint('done\n\n')

print('Baseline method: th_AccA')
print(f'best_acc: {b_acc:.4f}, best_prec: {b_prec:.4f}, auc_roc: {auc:.4f}')