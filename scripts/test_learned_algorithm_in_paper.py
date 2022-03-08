from autocfr.evaluator.vanilla_evaluator import VanillaEvaluator
from autocfr.utils import load_df, load_game_configs


def main():
    algo_names = ["DCFRPlus", "AutoCFR4", "AutoCFRS"]
    game_configs = load_game_configs(mode="test")
    evaluate(algo_names, game_configs)


def evaluate(algo_names, game_configs):
    evaluator = VanillaEvaluator(
        game_configs,
        algo_names,
        eval_freq=20,
        print_freq=100,
        num_iters=20000,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
