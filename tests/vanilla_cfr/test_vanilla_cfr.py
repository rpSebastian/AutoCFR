import pyspiel
from autocfr.vanilla_cfr.cfr import CFRSolver

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability


def test_cfr():
    game = pyspiel.load_game("kuhn_poker")
    solver = CFRSolver(game)

    for i in range(10):
        solver.iteration()

    conv = exploitability.exploitability(
        game,
        policy.tabular_policy_from_callable(game, solver.average_policy()),
    )
    assert conv == 0.0686987938171576
