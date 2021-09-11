from autocfr.cfr.cfr_algorithm import load_algorithm
import ray
import pytest
from autocfr.evolution import Evolution
from pathlib import Path
from autocfr.population import Agent
from autocfr.utils import load_game_configs


class FakeLogger:
    def info(self, str):
        pass


class TestEvolution:
    def get_init_evolution(self):
        evolution = Evolution(300, 25)
        evolution.initial_logger(FakeLogger())
        evolution.logger_info()
        game_configs = load_game_configs("test")
        evolution.init_game_config(game_configs)
        # evolution.init_tensorboard()
        return evolution

    def test_init_population_from_file(self):
        evolution = self.get_init_evolution()
        init_population_file = (
            Path(__file__).parent.parent / "models" / "population" / "test.pkl"
        )
        evolution.init_population(init_population_file=init_population_file)
        popu = evolution.population
        assert Agent.total_index == 2
        assert Agent.total_index == popu.max_agent_index + 1
        assert len(popu) == 2

    def atest_init_population_from_algorithms(self):
        evolution = self.get_init_evolution()
        init_algorithms_file = [
            Path(__file__).parent.parent / "models" / "algorithms" / "cfr.pkl",
            Path(__file__).parent.parent / "models" / "algorithms" / "linear_cfr.pkl",
            Path(__file__).parent.parent / "models" / "algorithms" / "dcfr.pkl",
            Path(__file__).parent.parent / "models" / "algorithms" / "cfr_plus.pkl",
        ]
        ray.init()
        evolution.init_population(init_algorithms_file=init_algorithms_file)
        ray.shutdown()
        popu = evolution.population
        assert pytest.approx(popu.max_score, 1e-3) == 1.000
        assert len(popu) == 4

    def test_init_population_from_empty(self):
        with pytest.raises(Exception, match="Must specify"):
            evolution = Evolution(300, 25)
            evolution.init_population()

    def test_copy_score_and_join_popu_direct(self):
        evolution = self.get_init_evolution()
        init_population_file = (
            Path(__file__).parent.parent / "models" / "population" / "test.pkl"
        )
        evolution.init_population(init_population_file=init_population_file)
        algorithm = load_algorithm("cfr")
        agent = Agent(algorithm, early_hurdle_score=0)
        agent.gen_hash_code()
        evolution.copy_score_and_join_popu_direct(agent)
        assert agent.ave_score == 0