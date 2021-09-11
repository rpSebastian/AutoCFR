from pathlib import Path

import pytest
from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.population import Agent, Population
from autocfr.standing import standing


class TestAgent():

    def test_score(self):
        game_configs = [
            {
                "long_name": "kuhn_poker",
                "game_name": "kuhn_poker",
                "params": {"players": 2},
                "iterations": 1000,
                "type": "small",
                "weight": 1,
                "max_score": 1.5
            },
            {
                "long_name": "liars_dice_1n_3s",
                "game_name": "liars_dice",
                "params": {"numdice": 1, "dice_sides": 2},
                "iterations": 1000,
                "type": "small",
                "weight": 1,
                "max_score": 1.5
            }
        ]
        algorithm = load_algorithm("cfr")
        agent = Agent(algorithm)
        for game_config in game_configs:
            score = standing.score(0.8, game_config)
            agent.set_score(game_config["long_name"], score, game_config["weight"])
        assert pytest.approx(agent.ave_score, 1e-3) == -2.161

    def test_dict(self):
        agent = Agent(None)
        agent_index = agent.index
        assert Agent.get_agent(agent_index) == agent


class TestPopulation():
    def test_save(self):
        Agent.total_index = 0
        cfr = load_algorithm("cfr")
        dcfr = load_algorithm("dcfr")
        agent_1 = Agent(cfr)
        agent_2 = Agent(dcfr)
        agent_1.set_score("Kuhn_Poker", 0, 1)
        agent_2.set_score("Kuhn_Poker", 1, 1)
        population = Population(300, 25)
        population.add_agent(agent_1)
        population.add_agent(agent_2)
        filename = Path(__file__).parent.parent / "models" / "population" / "test.pkl"
        Population.save(population, filename)

    def test_init_from_file(self):
        filename = Path(__file__).parent.parent / "models" / "population" / "test.pkl"
        population = Population.init_from_file(filename)
        assert Agent.total_index == 2
        assert Agent.total_index == population.max_agent_index + 1
        assert len(population) == 2

    def test_item(self):
        filename = Path(__file__).parent.parent / "models" / "population" / "test.pkl"
        population = Population.init_from_file(filename)
        assert pytest.approx(population[0].ave_score, 1e-3) == 0
        assert pytest.approx(population[0].early_hurdle_score, 1e-3) == 0
        assert pytest.approx(population.max_score, 1e-3) == 1
        assert pytest.approx(population.early_hurdle_threshold, 1e-3) == 0.5
