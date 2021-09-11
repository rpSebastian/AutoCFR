from autocfr.vanilla_evaluator import VanillaEvaluator
from autocfr.utils import load_nfg


def test_vanilla_evaluator():
    game_configs = load_nfg()
    algo_names = ["CFR", "DCFR"]
    evaluator = VanillaEvaluator(
        game_configs,
        algo_names,
        "test",
        num_iters=10,
        eval_freq=1,
        verbose=False
    )
    evaluator.evaluate()
    assert len(evaluator.df) == 110
