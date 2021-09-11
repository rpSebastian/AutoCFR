from autocfr.evaluator.vanilla_evaluator import VanillaEvaluator
from autocfr.utils import load_df, load_game_configs


def main():
    algo_names = ["CFR", "LinearCFR", "CFRPlus", "DCFR"]
    game_configs = load_game_configs(mode="train")
    save_name = "../baseline"
    evaluate(algo_names, save_name, game_configs)


def evaluate(algo_names, save_name, game_configs):
    evaluator = VanillaEvaluator(
        game_configs, algo_names, save_name, eval_freq=1, print_freq=1, num_iters=1000
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
