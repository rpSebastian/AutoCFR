import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from autocfr.utils import load_game_configs
from autocfr.evaluator.algorithm_evaluator import AlgorithmEvaluator

def load_best_algorithm(logid):
    json_file = Path(__file__).parent.parent / "logs" / str(logid) / "metrics.json"
    with json_file.open("r") as f:
        metrics = json.load(f)
    nas = metrics["new_ave_score"]
    df = pd.DataFrame(data=dict(step=nas["steps"], value=nas["values"]))
    df = df.sort_values(by="value", ascending=False)
    print(df.head(20))
    print(list(df.head(20).iloc[:, 0]))
    algorithm_index = int(df.iloc[0]["step"])
    algorithm_file = (
        Path(__file__).parent.parent
        / "logs"
        / str(logid)
        / "valid_algorithms"
        / "algorithms_{}.pkl".format(algorithm_index)
    )
    print("Log index: {}, The best algorithm index: {}".format(logid, algorithm_index))
    with algorithm_file.open("rb") as f:
        algorithm = pickle.load(f)
    return algorithm_index, algorithm


def main():
    args = get_args()
    algorithm_index, algorithm = load_best_algorithm(args.logid)
    game_configs = load_game_configs(mode="test")
    evaluator = AlgorithmEvaluator(
        game_configs,
        algorithm,
        "Algorithm_{}_{}".format(args.logid, algorithm_index),
        eval_freq=20,
        print_freq=100,
        num_iters=40000
    )
    evaluator.evaluate()


def get_args():
    parser = argparse.ArgumentParser(description="test learned algorithm")
    parser.add_argument("--logid", type=int, help="logid")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
