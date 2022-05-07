import numpy as np

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

# plot training loss
def plot_loss(loss, filename):
    plt.figure(figsize=(16, 4))
    plt.title("training loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.plot(loss)

    arg = np.argmin(loss, axis=0)
    val = loss[arg]
    plt.text(arg, val, '{:.4f}'.format(val))

    plt.savefig(filename)

def plot_result(preds, labels, filename):
    plt.figure(figsize=(16, 4))
    plt.title("anomaly")
    plt.plot(preds, label='predicted')
    plt.plot(labels, label='ground truth')

    plt.legend()
    plt.savefig(filename)

def plot_score(score, labels, filename):
    fig, ax1 = plt.subplots(figsize=(16, 4))
    ax1.plot(score, label='predicted', color='red')
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(labels, label='ground truth', color='orange')

    plt.legend()
    plt.savefig(filename)

def plot_heatmap(data, cmap=plt.cm.Blues, cheater=[], title='', filename='tmp.png'):
    l = len(data)
    fig, ax = plt.subplots(figsize=(6,5), dpi=400)
    im = ax.pcolor(data, cmap=cmap)
    fig.colorbar(im)

    ax.set_title(title, y=1.1)

    ax.set_xlabel('player_id')
    ax.set_ylabel('observer_id')

    ax.tick_params(direction="in")
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    for (j,i),d in np.ndenumerate(data):
        if str(i) in cheater:
            ax.text(i+0.5,j+0.5,f'{d:.4f}',ha='center',va='center',color='red')
        else:
            ax.text(i+0.5,j+0.5,f'{d:.4f}',ha='center',va='center')

    tick_pos = np.arange(0.5,l,1.0)
    ax.set_xticks(tick_pos, minor=False)
    ax.set_yticks(tick_pos, minor=False)
    ax.set_xticklabels(range(l), minor=False)
    ax.set_yticklabels(range(l), minor=False)
    plt.tight_layout()
    plt.gca().set_aspect('equal')

    fig.savefig(filename)

    plt.close()