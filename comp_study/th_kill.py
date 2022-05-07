# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *

# obtain the observed kill scores for each player
eval_trues, eval_scores = {}, {}
for game in sorted(glob('data_processed/*/*')):
    f_cheat = f'{game}/cheater'
    with open(f_cheat, 'r') as f:
        rows = f.readlines()
        cheater = [int(c[0]) for c in rows] if len(rows) else []

    f_player = f'{game}/player_id'
    with open(f_player, 'r') as f:
        rows = f.readlines()
        player_id = [int(c[0]) for c in rows]

    l = len(player_id)
    trues, scores = np.zeros((l,l)), np.zeros((l,l))

    for j,o in enumerate(player_id):
        log_event = f'{game}/obs_{o}/log_event_processed.csv'
        df_event = pd.read_csv(log_event)
        kill = df_event['event'] == 'dead'

        for k,p in enumerate(player_id):
            pl = df_event['dst'] == p
            n_kills = len(df_event[kill & pl])

            scores[j,k] = n_kills
            trues[j,k] = 1 if p in cheater else 0

    eval_trues[game] = trues
    eval_scores[game] = scores

# evaluate prediction performance
b_acc, b_prec, auc, _ = eval_preds(eval_trues, eval_scores)

eprint('done\n\n')

# print results
print('Baseline method: th_Kill')
print(f'best_acc: {b_acc:.4f}, best_prec: {b_prec:.4f}, auc_roc: {auc:.4f}')
