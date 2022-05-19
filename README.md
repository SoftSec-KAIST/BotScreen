# BotScreen

This repository contains implementation for paper ["BotScreen: Enabling Distributed and Real-Time Aimbot Detection"](https://).

## Dependencies

The code requires dependencies that can be installed using the `pip` environment:

```run
$ pip3 install -r requirements.txt
```

## Replicating Benchmark Results

### Downloading preprocessed data

We make pre-processed gameplay data available [here](https://) [TODO]. The compressed file from the link contains `data_processed/` folder, where the users can simply put them in the root directory.

While no longer functional, we include the script used in preprocessing in the repository ([`preprocess.py`](./preprocess.py)).

### Training SGRU models

For exact replication of the results in the paper, it is recommended to download weights of pre-trained SGRU models from [this link](https://drive.google.com/file/d/11c6YGBFEQC344Jvy9e9atz25pZ5K8nyR/view?usp=sharing). The compressed file (`best_models.zip`) in the link consists of 7 pre-trained model weights (`gru_k0.pt` ~ `gru_k6.pt`) and a JSON configuration file (`config.json`).

Simply make `trained_models/` directory and unzip `best_model.zip` in the directory to finish the setup.

For any user who wishes to train SGRU themselves, executing the following will train the models.
```train
$ make train [arguments]
```
See descriptions in [`makefile`](./makefile) and [`utils/__init__.py`](./utils/__init__.py) for different choices in arguments. Leaving argument empty will default to the arguments used in the paper.

Initially, the code will take a while (10-20 mins) in filtering necessary events among entire collected data (see Section 4.2 for details). After the initialization step, filtered events will be stored in `trained_models/_frame`, and be used in training and further evaluations.

<!--We make pre-filtered `_frame` available to download [here](https://).-->

### Evaluation

Once weights for SGRU are obtained, it can be evaluated by running the following command:
```eval
$ make eval [arguments]
```
The above command generates and saves ground truths and evaluation scores for each model. If the pre-trained model weights and configuration are used, running above will produce `eval_k0.pt` ~ `eval_k6.pt` in `trained_models`.

We make pre-evaluated data avaiable to download [here](https://drive.google.com/file/d/11a2vpY5Af_6_a43ZYIRmpYQYZ9U3F9TV/view?usp=sharing).

### Experiments

Scripts in `experiments/` and `comp_study` contain implementation for experiments appearing in the paper. Each code can be run as follows:
```exp
$ make [exp_name] [arguments]
```
For instance, if a user wishes to run script [`experiments/exp_bench.py`](./experiments/exp_bench.py), simply executing `make experiments/exp_bench` will run the experiment.

The following table summarizes experiments and corresponding results in the paper:

| exp_name | summary | outputs | section |
| - | - | - | - |
| `experiments/exp_bench` | Aimbot prediction performance | [`bench.tsv`](./bench/bench.tsv) | 6.2 |
| `experiments/exp_roc` | Plot ROC curves | [Figure 4](./figures/fig_04_roc.pdf) | 6.2 |
| `experiments/exp_std` | Difference btw. anomaly scores | [Figure 5](./figures/fig_05_std.pdf) | 6.4 |
| `experiments/stats_obs` | Stats of obs. rate (required before exp_obs) | [`data_loss/`](./data_loss) | 6.4 |
| `experiments/exp_obs` | Effect of obs. rate to accuracy | [Figure 6](./figures/fig_06_obsrate.pdf) | 6.4 |
| `experiments/exp_atk` | Effect of dishonest players | [Figure 7](./figures/fig_07_atk.pdf), [`bench_atk.tsv`](./bench/bench_atk.tsv) | 6.5 |
| `comp_study/th_vara` | Pred. performance of `th_vara` | - | 6.3 |
| `comp_study/th_acca` | Pred. performance of `th_acca` | - | 6.3 |
| `comp_study/th_kill` | Pred. performance of `th_kill` | - | 6.3 |
| `comp_study/ks_acca` | Pred. performance of `ks_acca` | - | 6.3 |

Note that in order for `experiments/exp_obs` to properly output expected results, running `experiments/stat_obs` must be preceeded. Additionally, `.tsv` files will be saved once `save_results=True` argument is passed alongside other arguments.

By default, we did not include any figures, data and benchmark results. The files in the outputs column will be accessible once the experiments are done.

## End-to-End Implementation

[TODO]

## Citation
If you find the provided code useful, please cite our work.
```
@inproceedings{
    choi2022botscreen,
    title={BotScreen: Enabling Distributed and Real-Time Aimbot Detection},
    author={Minyeop Choi and Gihyuk Ko and Sang Kil Cha},
    booktitle={TBD},
    year={2022}
}
```