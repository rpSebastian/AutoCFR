# AutoCFR: Learning to Design Counterfactual Regret Minimization Algorithms

> AutoCFR: Learning to Design Counterfactual Regret Minimization Algorithms <br>
> Hang Xu<sup>\*</sup> , Kai Li<sup>\*</sup>, Haobo Fu, Qiang Fu, Junliang Xing<sup>#</sup> <br>
> AAAI 2022 (Oral)


## Install AutoCFR

```bash
sudo apt install graphviz xdg-utils xvfb
conda create -n AutoCFR python==3.7.10
conda activate AutoCFR
pip install -e .
pytest tests
```

## Train AutoCFR

To easily run the code for training, we provide a unified interface. Each experiment will generate an experiment id and create a unique directory in `logs`. We use games implemented by [OpenSpiel](https://github.com/deepmind/open_spiel) [1].

```bash
python scripts/train.py
```

You can modify the configuration in the following ways:

1. Modify the operations. Some codes are from [Meta-learning curiosity algorithms](https://github.com/mfranzs/meta-learning-curiosity-algorithms)[2]. Specify your operations in `autocfr/program/operations.py` and specify a list of operations to use in `autocfr/generator/mutate.py`.
2. Modify the type and number of games used for training. Specify your game in `autocfr/utils.py:load_game_configs`.
3. By default, we learn from bootstrapping. If you want to learn from scratch, Set `init_algorithms_file` to `["models/algorithms/empty.pkl]` in `scripts/train.py`. 
4. Modify the hyperparameters. Edit the file `scripts/train.py`.
5. Train on distributed servers. Follow the instructions of [ray](https://docs.ray.io/en/master/cluster/cloud.html#cluster-private-setup) to setup your private cluster and set `ray.init(address="auto")` in `scripts/train.py`.

You can use Tensorboard to monitor the training process. 
```bash
tensorboard --logdir=logs
```

## Test learned algorithms by AutoCFR

Run the following script to test algorithms learned by AutoCFR. By default, we will test the algorithm with the highest score. `logid` is the generated unique experiment id. The results are saved in the folder `models/games`.
```bash
python scripts/test_learned_algorithm.py --logid={experiment id}
```

## Test learned algorithms in Paper
Run the following script to test learned algorithms in Paper, i.e., DCFR+, AutoCFR4, and AutoCFR8. The results are saved in the folder `models/games`.
```bash
python scripts/test_learned_algorithm_in_paper.py
```

## PokerRL

We use [PokerRL](https://github.com/EricSteinberger/PokerRL) [3] to test learned algorithms in HUNL Subgames.

### Install PokerRL

```bash
cd PokerRL
pip install -e .
tar -zxvf texas_lookup.tar.gz
```

### Test learned algorithms in Paper
Run the following script to test learned algorithms in Paper, i.e., DCFR+, AutoCFR4, and AutoCFRS. The results are saved in the folder `PokerRL/models/`.
```bash
python PokerRL/scripts/run_cfr.py --iters 20000 --game subgame3 --algo=DCFRPlus
python PokerRL/scripts/run_cfr.py --iters 20000 --game subgame3 --algo=AutoCFR4
python PokerRL/scripts/run_cfr.py --iters 20000 --game subgame3 --algo=AutoCFRS
```
## Citing
If you use AutoCFR in your research, you can cite it as follows:
```
@inproceedings{AutoCFR,
  title     = {AutoCFR: Learning to Design Counterfactual Regret Minimization Algorithms},
  author    = {Hang, Xu and Kai, Li and Haobo, Fu and Qiang, Fu and Junliang, Xing},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2022},
  pages     = {5244--5251}
}
```


## References

[1] Lanctot, M.; Lockhart, E.; Lespiau, J.-B.; Zambaldi, V.; Upadhyay, S.; P´erolat, J.; Srinivasan, S.; Timbers, F.; Tuyls, K.; Omidshafiei, S.; Hennes, D.; Morrill, D.; Muller, P.; Ewalds, T.; Faulkner, R.; Kram´ar, J.; Vylder, B. D.; Saeta, B.; Bradbury, J.; Ding, D.; Borgeaud, S.; Lai, M.; Schrittwieser, J.; Anthony, T.; Hughes, E.; Danihelka, I.; and Ryan-Davis, J. 2019. OpenSpiel: A Framework for Reinforcement Learning in Games. CoRR, abs/1908.09453.

[2] Alet, F.; Schneider, M. F.; Lozano-Perez, T.; and Kaelbling, L. P. 2019. Meta-learning curiosity algorithms. In International Conference on Learning Representations, 1–21.

[3] Steinberger, E. 2019. PokerRL. https://github.com/TinkeringCode/PokerRL.
