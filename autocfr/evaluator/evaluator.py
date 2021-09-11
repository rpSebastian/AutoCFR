import ray
import traceback

from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy

from autocfr.worker import Worker, VecWorker, GroupVecWorker
from autocfr.cfr.cfr_solver import CFRSolver
from autocfr.utils import load_game


def compute_conv(game, solver, exp_lower_limit=1e-12):
    conv = exploitability.exploitability(
        game,
        policy.tabular_policy_from_callable(game, solver.average_policy()),
    )
    conv = max(conv, exp_lower_limit)
    return conv


def evaluate_algorithm_run(
    game_config,
    algorithm,
    return_conv_records=False,
    conv_records_interval=10,
    verbose=True,
    exp_lower_limit=1e-12,
):
    conv_records = {}
    game = load_game(game_config)
    solver = CFRSolver(game, algorithm)
    for iters in range(game_config["iterations"]):
        solver.iteration()
        if return_conv_records and iters % conv_records_interval == 0:
            conv = compute_conv(game, solver, exp_lower_limit)
            conv_records[iters] = conv
            if verbose:
                print(
                    "iters: {}, game_name: {}, conv: {}".format(
                        iters, game_config["long_name"], conv
                    )
                )
    conv = compute_conv(game, solver, exp_lower_limit)
    return conv, conv_records


def evaluate_algorithm(
    game_config,
    algorithm,
    return_conv_records=False,
    conv_records_interval=10,
    verbose=True,
    exp_lower_limit=1e-12,
):
    try:
        conv, conv_records = evaluate_algorithm_run(
            game_config,
            algorithm,
            return_conv_records,
            conv_records_interval,
            verbose,
            exp_lower_limit,
        )
    except Exception as e:
        result = dict(status="fail", info=str(e), game_config=game_config, traceback=traceback.format_exc())
    else:
        result = dict(status="succ", conv=conv, game_config=game_config)
        if return_conv_records:
            result["conv_records"] = conv_records
    return result


class Evaluator(Worker):
    def __init__(
        self,
        index,
        return_conv_records=False,
        conv_records_interval=10,
        verbose=True,
        exp_lower_limit=1e-12,
    ):
        super().__init__(index)
        self.return_conv_records = return_conv_records
        self.conv_records_interval = conv_records_interval
        self.verbose = verbose
        self.exp_lower_limit = exp_lower_limit

    def run(self, task):
        game_config = task["game_config"]
        algorithm = task["algorithm"]
        agent_index = task["agent_index"]
        evaluate_result = evaluate_algorithm(
            game_config,
            algorithm,
            self.return_conv_records,
            self.conv_records_interval,
            self.verbose,
            self.exp_lower_limit,
        )

        result = self.get_result_dict(task)
        result.update(evaluate_result)
        result["agent_index"] = agent_index
        return result


class VecEvaluator(VecWorker):
    def __init__(
        self,
        num_evaluators,
        return_conv_records=False,
        conv_records_interval=10,
        verbose=True,
        exp_lower_limit=1e-12,
    ):
        kwargs = {
            "return_conv_records": return_conv_records,
            "conv_records_interval": conv_records_interval,
            "verbose": verbose,
            "exp_lower_limit": exp_lower_limit,
        }
        super().__init__(num_evaluators, Evaluator, **kwargs)

    def eval_algorithm(self, agent_index, algorithm, game_config):
        task = dict(
            agent_index=agent_index, algorithm=algorithm, game_config=game_config
        )
        self.add_task(task)

    def get_evaluating_result(self):
        return self.get_result()


class EvaluatorContainer(Worker):
    @ray.remote
    def run(
        task,
        return_conv_records=False,
        conv_records_interval=10,
        verbose=True,
        exp_lower_limit=1e-12,
    ):
        game_config = task["game_config"]
        algorithm = task["algorithm"]
        evaluate_result = evaluate_algorithm(
            game_config,
            algorithm,
            return_conv_records,
            conv_records_interval,
            verbose,
            exp_lower_limit,
        )
        result = {
            "worker_index": task["worker_index"],
            "group_index": task["group_index"],
            "agent_index": task["agent_index"],
        }
        result.update(evaluate_result)
        return result


class GroupVecEvaluator(GroupVecWorker):
    def __init__(
        self,
        num_evaluators,
        return_conv_records=False,
        conv_records_interval=10,
        verbose=True,
        exp_lower_limit=1e-12,
    ):
        kwargs = {
            "return_conv_records": return_conv_records,
            "conv_records_interval": conv_records_interval,
            "verbose": verbose,
            "exp_lower_limit": exp_lower_limit,
        }
        super().__init__(num_evaluators, EvaluatorContainer, **kwargs)

    def eval_algorithm_parallel(self, agent_index, algorithm, game_configs):

        tasks = []
        for game_config in game_configs:
            task = dict(
                agent_index=agent_index, algorithm=algorithm, game_config=game_config
            )
            tasks.append(task)
        self.add_tasks(tasks)

    def get_evaluating_result(self):
        return self.get_result()
