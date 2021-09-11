import ray
from autocfr.exp import ex
from autocfr.evolution import Evolution
from autocfr.utils import load_game_configs
import numpy as np


@ex.config
def config():
    seed = 0
    population_size = 300
    tournament_size = 25
    num_evaluators = 20
    num_generators = 5

    print_freq = 20  
    save_freq = 1000  
    agent_save_freq = 100  

    init_algorithms_file = [
        "models/algorithms/cfr.pkl",
        "models/algorithms/cfr_plus.pkl",
        "models/algorithms/dcfr.pkl",
        "models/algorithms/linear_cfr.pkl",
    ] 
    check_program = True
    programs_max_length = [8, 16, 6]
    crossover_prob = 0.1
    early_hurdle = True
    early_hurdle_iters = 100
    exp_lower_limit = 1e-12
    game_configs = load_game_configs(mode="train")

@ex.automain
def main(_log):
    ray.init()
    # ray.init(address="auto")
    evolution = Evolution()
    evolution.initial()
    evolution.start_evolve()
    ray.shutdown()
