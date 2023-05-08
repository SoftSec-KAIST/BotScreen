# BotScreen

This repository contains implementation for paper ["BotScreen: Trust Everybody, but Cut the Aimbots Yourself"](https://).

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

For exact replication of the results in the paper, it is recommended to download weights of pre-trained SGRU models from [this link](https://drive.google.com/file/d/11e4lnYkQ2km_DsvI5_MPMclX_yITyqcZ/view?usp=sharing). The compressed file (`best_models.zip`) in the link consists of 7 pre-trained model weights (`gru_k0.pt` ~ `gru_k6.pt`) and a JSON configuration file (`config.json`).

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

We make pre-evaluated data avaiable to download [here](https://drive.google.com/file/d/11cj8PWcVw0HWeka1Ny79Dp7Qf2OWkpV5/view?usp=sharing).

### Experiments

Scripts in `experiments/` and `comp_study/` contain implementation for experiments appearing in the paper. Each code can be run as follows:
```exp
$ make [exp_name] [arguments]
```
For instance, if a user wishes to run script [`experiments/exp_bench.py`](./experiments/exp_bench.py), simply executing `make experiments/exp_bench` will run the experiment.

The following table summarizes experiments and corresponding results in the paper:

| exp_name | summary | outputs | section |
| - | - | - | - |
| `experiments/exp_bench` | Aimbot prediction performance | [`bench.tsv`](./bench/bench.tsv) | 5.2.1 |
| `experiments/exp_roc` | Plot ROC curves | Figure 4 | 5.2.2 |
| `experiments/exp_std` | Difference btw. anomaly scores | Figure 7 | 5.5.1 |
| `experiments/stats_obs` | Stats of obs. rate (required before exp_obs) | Figure 8 | 5.5.2 |
| `experiments/exp_obs` | Effect of obs. rate to accuracy | Figure 8 | 5.5.2 |
| `comp_study/th_vara` | Pred. performance of `th_VarA` | - | 5.4 |
| `comp_study/th_acca` | Pred. performance of `th_AccA` | - | 5.4 |
| `comp_study/th_kill` | Pred. performance of `th_Kill` | - | 5.4 |
| `comp_study/ks_acca` | Pred. performance of `ks_AccA` | - | 5.4 |
| `comp_study/os_cac` | Pred. performance of `os_CAC` | - | 5.4 |
| `comp_study/os_lac` | Pred. performance of `os_LAC` | - | 5.4 |
| `comp_study/os_smac` | Pred. performance of `os_SMAC` | - | 5.4 |
| `comp_study/history` | History-based detection accuracies | Table 10 | Appendix D |

Note that in order for `experiments/exp_obs` to properly output expected results, running `experiments/stat_obs` must be preceeded. Additionally, `.tsv` files will be saved once `save_results=True` argument is passed alongside other arguments.

By default, we did not include any figures, data and benchmark results. The files in the outputs column will be accessible once the experiments are done.

## Demo Implementation

We implement the demo of BotScreen on [Osiris](https://github.com/danielkrupinski/Osiris/tree/5af83362a69367fe3ed441a5e6218762a8196372), open-source memory-based cheat for CS:GO.

In this implementation, we apply the best accuracy model from game-based evaluation for the detection model.

## Citation
If you find the provided code useful, please cite our work.
```
@inproceedings{
    botscreen,
    title={BotScreen: Trust Everybody, but Cut the Aimbots Yourself},
    author={Minyeop Choi and Gihyuk Ko and Sang Kil Cha},
    booktitle={TBD},
    year={TBD}
}
```