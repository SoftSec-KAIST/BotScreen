# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
e = exp_from_arguments()        # get arguments

# placeholders for stats
fig, ax = plt.subplots()
tprs, aucs = [], []
mean_fpr = np.linspace(0, 1, 100)

# for each split
for i,test_games in enumerate(e.splits):

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

    # aggregated true labels and error scores
    t_agg = np.concatenate([get_votes(t)[0] for t in eval_trues.values()])
    s_agg = sum([get_median(s) for s in eval_scores.values()], [])

    # roc curve for prediction
    v = RocCurveDisplay.from_predictions(
        t_agg,
        s_agg,
        name=f"ROC fold {i}",
        alpha=0.3,
        lw=1,
        ax=ax
    )
    i_tpr = np.interp(mean_fpr, v.fpr, v.tpr)
    i_tpr[0] = 0.0
    tprs.append(i_tpr)
    aucs.append(v.roc_auc)

    eprint('done\n\n')

# chance and boundary plots for aggregate prediction
ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=1.5,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
)
ax.legend(loc="lower right")
os.makedirs('figures',exist_ok=True)
fig.savefig('figures/fig_04_roc.pdf')