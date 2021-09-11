import pickle
import random
import traceback
from pathlib import Path

import numpy as np

from autocfr.program.executor import AlgorithmExecutionError
from autocfr.program.program_types import nptype


class ProgramCheck():
    def __init__(self):
        self.init_random_sample()

    def load_samples(self, filename):
        path = Path(__file__).parent.parent.parent / "models" / filename
        with path.open("rb") as f:
            random_samples = pickle.load(f)
        return random_samples

    def init_random_sample(self):
        self.random_samples = []
        filenames = [
            "random_inputs.pkl",
            "algo_inputs_a.pkl",
        ]
        for filename in filenames:
            self.random_samples.extend(self.load_samples(filename))

    def program_check(self, algorithm):
        try:
            self.program_check_run(algorithm)
        except AlgorithmExecutionError as e:
            result = dict(status="fail", info=str(e), traceback=traceback.format_exc())
        else:
            result = dict(status="succ")
        return result

    def program_check_run(self, algorithm):
        for input_values_of_names in self.random_samples:
            algorithm.execute(input_values_of_names)


def generate_random_inputs():
    samples = []
    for _ in range(20):
        na = 20
        input_values_of_names = {
            "ins_regret": np.random.uniform(-100, 100, na),
            "reach_prob": nptype(np.random.uniform(0, 1)),
            "iters": nptype(np.random.randint(1, 1000)),
            "cumu_regret": np.random.uniform(-1000, 1000, size=na),
            "strategy": np.random.uniform(0, 1, size=na),
            "cumu_strategy": np.random.uniform(0, 1000, size=na),
        }
        samples.append(input_values_of_names)
    for _ in range(20):
        na = 20
        input_values_of_names = {
            "ins_regret": np.random.uniform(-1, 1, na),
            "reach_prob": nptype(np.random.uniform(0, 1)),
            "iters": nptype(np.random.randint(1, 10)),
            "cumu_regret": np.random.uniform(-10, 10, size=na),
            "strategy": np.random.uniform(0, 1, size=na),
            "cumu_strategy": np.random.uniform(0, 10, size=na),
        }
        samples.append(input_values_of_names)
    path = Path(__file__).parent.parent.parent / "models" / "random_inputs.pkl"
    with path.open("wb") as f:
        pickle.dump(samples, f)


program_check = ProgramCheck()
