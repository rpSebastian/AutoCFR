import time
import pytest
from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.cfr.cfr_solver import CFRSolver
from autocfr.evaluator import compute_conv, evaluate_algorithm, Evaluator, VecEvaluator, GroupVecEvaluator
from autocfr.utils import load_game


def get_game_config():
    game_config = dict(
        long_name="kuhn_poker",
        game_name="kuhn_poker",
        params=dict(players=2),
        iterations=10,
    )
    return game_config


def test_compute_conv():
    game_config = get_game_config()
    game = load_game(game_config)
    algorithm = load_algorithm("cfr")
    solver = CFRSolver(game, algorithm)
    conv = compute_conv(game, solver)
    assert pytest.approx(conv, 1e-3) == 0.45833


def test_evaluate_algorithm():
    game_config = get_game_config()
    algorithm = load_algorithm("cfr")
    result = evaluate_algorithm(game_config, algorithm)
    assert result["status"] == "succ"
    assert pytest.approx(result["conv"], 1e-3) == 0.06869
    assert result["game_config"]["long_name"] == "kuhn_poker"

    algorithm = load_algorithm("cfr_error")
    result = evaluate_algorithm(game_config, algorithm)
    assert result["status"] == "fail"


def test_evaluator():
    evaluator = Evaluator(0)
    game_config = get_game_config()
    algorithm = load_algorithm("cfr")
    task = {"agent_index": 1, "algorithm": algorithm, "game_config": game_config}
    result = evaluator.run(task)
    assert result["status"] == "succ"
    assert result["agent_index"] == 1
    assert result["worker_index"] == 0
    assert pytest.approx(result["conv"], 1e-3) == 0.06869


def atest_vec_evaluator():
    import ray
    ray.init()
    vec_evaluator = VecEvaluator(1)
    game_config = get_game_config()
    algorithm = load_algorithm("cfr")
    vec_evaluator.eval_algorithm(
        1, algorithm, game_config
    )
    for i in range(3):
        time.sleep(1)
        result = vec_evaluator.get_evaluating_result()
        if result is not None:
            assert result["status"] == "succ"
            assert result["agent_index"] == 1
            assert result["game_config"]["long_name"] == "kuhn_poker"
            assert pytest.approx(result["conv"], 1e-3) == 0.06869
    ray.shutdown()


def atest_group_vec_evaluator():
    import ray
    ray.init()
    vec_evaluator = GroupVecEvaluator(2)
    game_configs = [
        get_game_config(),
        get_game_config(),
    ]
    algorithm = load_algorithm("cfr")
    vec_evaluator.eval_algorithm_parallel(
        1, algorithm, game_configs
    )
    algorithm = load_algorithm("cfr_error")
    vec_evaluator.eval_algorithm_parallel(
        1, algorithm, game_configs
    )
    for i in range(3):
        time.sleep(1)
        result = vec_evaluator.get_evaluating_result()
        if result is not None:
            print(result)
    ray.shutdown()
