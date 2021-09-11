# AutoCFR
This is the code for "AutoCFR: Learning to Design Counterfactual Regret Minimization Algorithms".

## Install AutoCFR

```bash
sudo apt install graphviz xdg-utils xvfb
conda create -n autocfr python==3.7
conda activate autocfr
pip install -e .
pytest tests
```

## Train AutoCFR

To easily run the code for training, we provide a unified interface. Each experiment will generate an experiment id and create a unique directory in `logs`. 
```bash
python scripts/train.py
```

You can modify the configuration in the following ways:

1. Modify the operations. Specify your operations in `autocfr/program/operations.py` and specify a list of operations to use in `autocfr/generator/mutate.py`.
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
Run the following script to test learned algorithms in Paper, i.e., DDCFR, AutoCFR4, and AutoCFR8. The results are saved in the folder `models/games`.
```bash
python scripts/test_learned_algorithm_in_paper.py
```
