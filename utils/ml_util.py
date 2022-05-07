from tqdm import trange
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import sys, torch

from more_itertools import *

# model: stacked gru
class sgru(torch.nn.Module):
    def __init__(self, n_tags, n_hidden, n_layer):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=n_hidden,
            num_layers=n_layer,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = torch.nn.Linear(n_hidden * 2, n_tags)

    def forward(self, x):
        x = x.transpose(0, 1)   # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        #return x[0] + out
        return out

# sliding window dataset
class AimDataset(Dataset):
    def __init__(self, data, data_cols, stride=1, win_size=90, ts_col='timestamp', atk_col=None):
        self.ts = np.array(data[ts_col])
        self.tag_values = np.array(data[data_cols], dtype=np.float32)
        self.valid_idxs = []
        self.win_size = win_size
        for L in range(len(self.ts) - self.win_size + 1):
        #for L in trange(len(self.ts) - self.win_size + 1, desc="inference", ncols=100):
            R = L + self.win_size - 1
            if self.ts[R] - self.ts[L] == self.win_size - 1:
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]

        self.n_idxs = len(self.valid_idxs)
        if atk_col is not None:
            self.attacks = np.array(data[atk_col], dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.win_size - 1
        first = self.tag_values[i]
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + self.win_size - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + self.win_size - 1] - first)
        item["answer"] = torch.from_numpy(self.tag_values[last] - first)
        return item


# early stopping
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


### training
def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()

    epochs = trange(n_epochs, desc="training", ncols=100)
    best = {"loss": sys.float_info.max}
    loss_history = []
    for e in epochs:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            guess = model(given)
            answer = batch["answer"].cuda()
            loss = loss_fn(answer, guess)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        loss_history.append(epoch_loss)
        epochs.set_postfix_str(f"loss: {epoch_loss}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1

    return best, loss_history


### inference
def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer - guess).cpu().numpy())
            try:
                att.append(np.array(batch["attack"]))
            except:
                att.append(np.zeros(batch_size))

    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )


def process_score(ts, score, method='cnt'):
    new_ts, new_score = [], []

    if method == 'cnt':
        for g in consecutive_groups(zip(ts, score), lambda x: x[0]):
            group = list(g)

            idx = np.argmax([s for _,s in group])
            new_ts.append(group[idx][0])
            new_score.append(group[idx][1])

    if method == 'dur_per':
        new_ts = ts
        for g in consecutive_groups(zip(ts, score), lambda x: x[0]):
            group = list(g)
            mx_score = max([s for _,s in group])
            new_score.extend([mx_score]*len(group))

    if method == 'dur':
        new_ts, new_score = ts, score

    # ts and scores have same length
    assert len(new_ts) == len(new_score)

    return np.array(new_ts), np.array(new_score)


def process_dist(ts, dist, method='max'):
    new_ts, new_dist, new_score = [], [], []

    score = np.linalg.norm(dist, ord=1.0, axis=1)/dist.shape[1]

    for g in consecutive_groups(zip(ts, score, dist), lambda x: x[0]):
        group = list(g)

        if method=='max':
            idx = np.argmax([s for _,s,_ in group])
            new_ts.append(group[idx][0])
            new_score.append(group[idx][1])
            new_dist.append(group[idx][2])

        elif method=='stdev':
            idx = np.argmax([s for _,s,_ in group])
            scores = [s for _,s,_ in group]
            new_ts.append(group[idx][0])
            new_score.append(np.std(scores))
            new_dist.append(group[idx][2])

    return np.array(new_ts), np.array(new_dist), np.array(new_score)
