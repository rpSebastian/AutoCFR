import ray
import pytest
from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.generator.generator import Generator
from autocfr.generator.generator import VecGenerator


def test_generator():
    generator = Generator(1, crossover_prob=1)
    cfr = load_algorithm("cfr")
    dcfr = load_algorithm("dcfr")
    early_hurdle_threshold = 0
    task = {
        "parent_algorithm_A": cfr,
        "parent_algorithm_B": dcfr,
        "early_hurdle_threshold": early_hurdle_threshold,
    }
    result = generator.run(task)
    for agent_info in result["agent_infos"]:
        assert agent_info["early_hurdle_score"] > early_hurdle_threshold - 1e-6

    generator = Generator(1, crossover_prob=0)
    cfr = load_algorithm("cfr")
    dcfr = load_algorithm("dcfr")
    task = {
        "parent_algorithm_A": cfr,
        "parent_algorithm_B": dcfr,
        "early_hurdle_threshold": early_hurdle_threshold
    }
    result = generator.run(task)
    assert len(result["agent_infos"]) == 1

    generator = Generator(1, crossover_prob=0)
    cfr = load_algorithm("cfr")
    dcfr = load_algorithm("dcfr")
    task = {
        "parent_algorithm_A": cfr,
        "parent_algorithm_B": dcfr,
        "early_hurdle_threshold": 1
    }
    result = generator.run(task)
    assert len(result["agent_infos"]) == 0


def atest_vec_generator():
    ray.init()
    vec_generator = VecGenerator(num_generators=2)
    cfr = load_algorithm("cfr")
    dcfr = load_algorithm("dcfr")
    vec_generator.gen_agent_infos(cfr, dcfr, early_hurdle_threshold=-1)
    for i in range(2):
        import time
        time.sleep(1)
        agent_infos = vec_generator.get_agent_infos()
        if len(agent_infos) > 0:
            print(agent_infos)
            assert len(agent_infos) > 0
    ray.shutdown()
