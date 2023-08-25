# BotScreen

This repository contains the prototype implementation of [BotScreen](https://www.usenix.org/conference/usenixsecurity23/presentation/choi), which is an aimbot detection system for first-person shooter games.

## Dependencies

Our implementation is written in Python 3. We recommend using Python version 3.10, as it is the version we have developed and tested our codes in. To run the code, first install the following dependencies:
```run
$ pip3 install -r requirements.txt
```

## Replicating Benchmark Results

### Downloading preprocessed data

We make pre-processed game-play data available [here](https://doi.org/10.5281/zenodo.8058051). The compressed file from the link contains `data_processed/` folder, where the users can simply put them in the root directory.

Due to the privacy reasons, we cannot share the raw data, but we include the script used in our preprocessing step in the repository ([`preprocess.py`](./preprocess.py)) for future reference.

### Training SGRU models

For any user who wishes to train SGRU themselves, executing the following will train the models.
```train
$ make train [arguments]
```
See descriptions in [`makefile`](./makefile) and [`utils/__init__.py`](./utils/__init__.py) for different argument choices. Leaving argument empty will default to the arguments used in the paper.

Initially, the code will take a while (10-20 mins) in filtering necessary events among entire collected data (see Section 4.2.1 for details). After the initialization step, filtered events will be stored in `trained_models/_frame`, and be used in training and further evaluations.

When the training is done, the trained weights will be saved `gru_k0.pt` ~ `gru_k6.pt` in `trained_models`.

### Evaluation

Once weights for SGRU are obtained, it can be evaluated by running the following command:
```eval
$ make eval [arguments]
```
The above command generates and saves ground truths and evaluation scores for each model. If the pre-trained model weights and configuration are used, running above will produce `eval_k0.pt` ~ `eval_k6.pt` in `trained_models`.

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
| `experiments/exp_roc` | Plot ROC curves | Figure 4 | 5.2.1 |
| `experiments/exp_std` | Difference btw. anomaly scores | Figure 7 | 5.5.1 |
| `experiments/stat_obs` | Stats of obs. rate (required before exp_obs) | Figure 8 | 5.5.2 |
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

## Artifacts

For exact replication of the results in the paper, it is recommended to download weights of pre-trained SGRU models from [this link](https://doi.org/10.5281/zenodo.8058051).

The `trained_models/` directory consists of 7 pre-trained model weights and a JSON configuration file (`config.json`). We also include pre-evaluated data in `trained models/` for a faster reproduction of the experiments in our paper.

## Data Collection & Demo Implementation

We implement game-play data collection tool and demo of BotScreen on [Osiris](https://github.com/danielkrupinski/Osiris/tree/5af83362a69367fe3ed441a5e6218762a8196372), an open-source memory-based cheat for CS:GO.

The following table summarizes major modifications done on Osiris:

| Path | Description |
| - | - |
| `Demo/Source/Hacks/Dump.cpp` | Game data extraction, Implementation of `Data Manager` |
| `Demo/Source/EventListener.cpp` | Additional event listener for BotScreen |
| `Demo/Source/Hooks.cpp` | Send `data manager` game data from hooked function `createMove` |
| `Demo/SGX_MDL/SGX_MDL.c` | Implementation of `Detector` |
| `Demo/SGX_MDL/state.h` | Weights for detection model |

By changing the pre-defined macro value `DUMP_LOG` to 1, Osiris will save two log files in `DUMP_PATH`, as the game proceeds; `log_event.csv` which logs game events such as 'fire', 'hit', or 'death', and `log_player.csv` which contains information of the players who is currently being rendered (i.e., visible) in the client's screen.

For detection model, we apply the best accuracy model from game-based evaluation for the detection model.

Once you inject BotScreen into CS:GO client and start the game, BotScreen will send the detection report to the server defined in `Demo/Source/Hacks/Dump.cpp`.

Note that this demo works only on Windows, and it was tested against CS:GO client build id 10852507. It might crash on another version of CS:GO client. In such cases, you should manually adopt a proper version of Osiris.

## Citation

If you find the provided code useful, please cite our work.
```
@inproceedings{botscreen,
    title={{BotScreen}: Trust Everybody, but Cut the Aimbots Yourself},
    author={Minyeop Choi and Gihyuk Ko and Sang Kil Cha},
    booktitle = {32nd USENIX Security Symposium (USENIX Security 23)},
    year = {2023},
}
```
