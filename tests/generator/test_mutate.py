import random
import numpy as np
from autocfr.generator.mutate import mutate
from autocfr.cfr.cfr_algorithm import load_algorithm


def test_mutate_algorithm():
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    algorithm = load_algorithm("cfr")
    mutate.mutate_algorithm(algorithm)
