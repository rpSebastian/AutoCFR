import math
import pandas as pd
from pathlib import Path


class Standing:
    def __init__(self):
        self.load_baseline_score()

    def load_baseline_score(self):
        csv_file = Path(__file__).parent.parent / "models" / "baseline.csv"
        self.baseline_score = pd.read_csv(csv_file)

    def score(self, exp, game_config):
        """将某游戏的利用率转换为评分

        Args:
            exp (float): 程序利用率

        Returns:
            score (float): 程序评分
        """
        iters = game_config["iterations"]
        game_name = game_config["long_name"]
        cfr_exp = self.get_exp("CFR", game_name, iters)
        dcfr_exp = self.get_exp("DCFR", game_name, iters)
        score = (math.log(cfr_exp) - math.log(exp)) / (
            math.log(cfr_exp) - math.log(dcfr_exp)
        )
        score = min(score, game_config["max_score"])
        return score

    def get_exp(self, algorithm_name, game_name, iters):
        df = self.baseline_score
        exp = df[
            (df.algorithm_name == algorithm_name) &
            (df.game_name == game_name) &
            (df.step == iters)
        ].iloc[0]["exp"]
        return exp


standing = Standing()
