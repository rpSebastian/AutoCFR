import random

import numpy as np
from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.generator.cross import cross


def test_early_hurdle_score():
    cfr = load_algorithm("cfr")
    dcfr = load_algorithm("dcfr")
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    cross.cross_algorithms(cfr, dcfr)
