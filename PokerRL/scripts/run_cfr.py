# from PokerRL._.CrayonWrapper import CrayonWrapper
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PokerRL.cfr import VanillaCFR, CFRPlus, LinearCFR, DCFR, DCFRPlus, AutoCFRS, AutoCFR4
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLHoldemSubGame3, DiscretizedNLHoldemSubGame4
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

parser = argparse.ArgumentParser(description="Run CFR in games")
parser.add_argument("--iters", type=int, help="iterations")
parser.add_argument("--algo", type=str, help="algo names")
parser.add_argument("--game", type=str, help="game names")

parser.add_argument("--save", action="store_true", default=False, help="game names")

args = parser.parse_args()

n_iterations = args.iters
algo_name = args.algo
game_name = args.game
name = "{}_{}".format(algo_name, game_name)

algo_dict = {
    "CFR": VanillaCFR,
    "CFRPlus": CFRPlus,
    "LinearCFR": LinearCFR,
    "DCFR": DCFR,
    "DCFRPlus": DCFRPlus,
    "AutoCFRS": AutoCFRS,
    "AutoCFR4": AutoCFR4,
}
game_dict = {
    "subgame3": DiscretizedNLHoldemSubGame3,
    "subgame4": DiscretizedNLHoldemSubGame4,
}

# Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
chief = ChiefBase(t_prof=None)
cfr = algo_dict[algo_name](
    name=name,
    game_cls=game_dict[game_name],
    agent_bet_set=bet_sets.B_3,
    other_agent_bet_set=bet_sets.B_2,
    chief_handle=chief,
)
c = []
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), "start run: ", name)

steps = []
convs = []

for step in range(1, n_iterations + 1):
    cfr.iteration()
    conv = cfr.expl / 1000
    print(
        time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),
        "Iteration: {:>4d} exp: {:>12.5f}".format(step, conv),
    )
    if step == 1:
        steps.append(0)
        convs.append(conv)
    steps.append(step)
    convs.append(conv)
    if args.save:
        # print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()), "saving")
        file = (
            Path(__file__).parent.parent.parent
            / "models"
            / game_name
            / "{}_{}.csv".format(algo_name, game_name)
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        algo_df = pd.DataFrame(
            data=dict(
                step=steps,
                exp=convs,
            )
        )
        algo_df["game_name"] = game_name
        algo_df["algorithm_name"] = algo_name
        algo_df["log_exp"] = np.log(algo_df["exp"]) / np.log(10)
        algo_df.to_csv(file, index=False)
