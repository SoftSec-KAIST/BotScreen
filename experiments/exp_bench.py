# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

# placeholder for result
res = []

# for each split
for i,test_games in enumerate(e.splits):

    # run on single split
    if e.split >= 0:
        if i != e.split:
            continue

    eprint(f'### split {i} ###' +'\n')

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

    # evaluate prediction performance
    res.append(eval_preds(eval_trues, eval_scores))

    eprint('done\n\n')


# save bench results
if e.save_results:
    os.makedirs('bench',exist_ok=True)
    if not os.path.exists('bench/bench.tsv'):
        with open('bench/bench.tsv','w') as f:
            f.write('\t'.join(['n_split', 'split', 'win_size', 'duration', 'seed',
                               'best_acc', 'best_prec', 'auc_roc', 'n_player']))
            f.write('\n')

    for i,row in enumerate(res):
        info = map(str,[e.n_split,i,e.win_size,e.duration,e.seed])
        stats = map(lambda s: f'{s:.4f}', row[:3])

        with open('bench/bench.tsv','a') as f:
            f.write('\t'.join(info))
            f.write('\t')
            f.write('\t'.join(stats))
            f.write('\t' + str(int(row[-1])))
            f.write('\n')


# weighted average for the experiment
res = np.array(res)
b_acc, b_prec, auc = np.average(res[:,:3], axis=0, weights=res[:,-1])
print('Botscreen:')
print(f'best_acc: {b_acc:.4f}, best_prec: {b_prec:.4f}, auc_roc: {auc:.4f}')