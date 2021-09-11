from autocfr.evaluator.vanilla_evaluator import VanillaEvaluator
from autocfr.utils import load_df, load_game_configs


def main():
    algo_names = ["DDCFR", "AutoCFR4", "AutoCFRS"]
    game_configs = load_game_configs(mode="test")
    save_name = "../baseline"
    evaluate(algo_names, save_name, game_configs)


def evaluate(algo_names, save_name, game_configs):
    evaluator = VanillaEvaluator(
        game_configs,
        algo_names,
        save_name,
        eval_freq=20,
        print_freq=100,
        num_iters=40000,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
