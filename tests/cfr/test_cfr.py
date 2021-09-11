import pytest
from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.cfr.cfr_solver import CFRSolver
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import discounted_cfr


class TestCFRSolver:
    def test_classical_algorithm_with_official(self):
        game = pyspiel.load_game("kuhn_poker")
        official_dict = {
            "cfr": cfr.CFRSolver,
            "cfr_plus": cfr.CFRPlusSolver,
            "linear_cfr": discounted_cfr.LCFRSolver,
            "dcfr": discounted_cfr.DCFRSolver
        }
        for algorithm_name, official_solver_class in official_dict.items():
            official_solver = official_solver_class(game)
            algorithm = load_algorithm(algorithm_name)
            algorithm_solver = CFRSolver(game, algorithm)
            for _ in range(3):
                official_solver.evaluate_and_update_policy()
                algorithm_solver.iteration()
            official_conv = exploitability.exploitability(
                game,
                policy.tabular_policy_from_callable(game, official_solver.average_policy()),
            )
            algorithm_conv = exploitability.exploitability(
                game,
                policy.tabular_policy_from_callable(game, algorithm_solver.average_policy()),
            )
            assert pytest.approx(algorithm_conv, 1e-3) == official_conv

    def test_empty(self):
        game = pyspiel.load_game("kuhn_poker")
        algorithm = load_algorithm("empty")
        algorithm_solver = CFRSolver(game, algorithm)
        for _ in range(3):
            algorithm_solver.iteration()
        algorithm_conv = exploitability.exploitability(
            game,
            policy.tabular_policy_from_callable(game, algorithm_solver.average_policy()),
        )
        assert pytest.approx(algorithm_conv, 1e-3) == 0.4583
