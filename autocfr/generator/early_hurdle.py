import traceback
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from autocfr.cfr.cfr_solver import CFRSolver
from autocfr.standing import standing
from autocfr.utils import load_game


class EarlyHurdle:
    def early_hurdle_score(
        self, program, early_hurdle_iters=100, exp_lower_limit=1e-12
    ):
        try:
            score = self.early_hurdle_score_run(
                program, early_hurdle_iters, exp_lower_limit
            )
        except Exception as e:
            result = dict(status="fail", info=str(e), traceback=traceback.format_exc())
        else:
            result = dict(status="succ", score=score)
        return result

    def early_hurdle_score_run(self, program, early_hurdle_iters, exp_lower_limit):
        game_config = {
            "long_name": "kuhn_poker",
            "game_name": "kuhn_poker",
            "params": {"players": 2},
            "iterations": early_hurdle_iters,
            "type": "small",
            "max_score": 1.5
        }
        game = load_game(game_config)
        solver = CFRSolver(game, program)
        for i in range(game_config["iterations"]):
            solver.iteration()
        conv = exploitability.exploitability(
            game,
            policy.tabular_policy_from_callable(game, solver.average_policy()),
        )
        conv = max(conv, exp_lower_limit)
        score = standing.score(conv, game_config)
        return score


early_hurdle = EarlyHurdle()
