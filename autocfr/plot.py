import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xpoker.utils import load_df

plt.rc("pdf", fonttype=42)
plt.rc("ps", fonttype=42)


class PlotAlgorithmCompare:
    def __init__(
        self,
        game_name,
        legend=True,
        iterations=20000,
        print_freq=20,
        save_dir="performance",
    ):
        print("Plot {}".format(game_name))
        self.save_dir = save_dir
        self.legend = legend
        if self.legend:
            self.save_name += "_legend"
        self.tick_list = list(range(self.tick_min, self.tick_max + 1))
        self.ymin = self.tick_min
        self.ymax = self.tick_max
        self.format = "png"
        self.game_name = game_name
        self.algorithm_names = [
            "CFR",
            "CFRPlus",
            "LinearCFR",
            "DCFR",
            # "DDCFR",
            "DCFRPlus",
            # "DCFRPlusTwo",
            # "AutoCFR4",
            # "AutoCFRS",
        ]
        self.print_freq = print_freq
        self.iterations = iterations
        self.transparent = False
        self.chinese = False
        self.font = matplotlib.font_manager.FontProperties(
            fname="C:\\Windows\\Fonts\\SimHei.ttf"
        )

    def run(self):
        self.set_font_size()
        df = self.load_algorithm()
        # df = self.filter_algorithm(df)
        df = self.filter_freq(df)
        df = self.filter_iterations(df)
        df = df.reset_index()
        self.plot_exp_compare_in_one_game(df, self.game_name)

    def plot_exp_compare_in_one_game(self, df, game_name):
        fig = plt.figure(figsize=(6, 4))
        df = self.filter_game(df, [game_name])
        sns.set_style("darkgrid")

        for algorithm_name in self.algorithm_names:
            self.plot_one_algorithm_exp(df, algorithm_name)
        if self.legend:
            plt.legend()
        plt.title(self.title_name)
        if self.chinese:
            plt.xlabel("迭代次数", fontproperties=self.font)
            plt.ylabel("可利用度(与纳什均衡策略的距离)", fontproperties=self.font)
        else:
            plt.xlabel("Iterations")
            plt.ylabel("Exploitability")
        self.set_xticks()
        self.set_yticks()
        self.set_lim()
        path = Path("images/{}/{}/{}.{}".format(self.save_dir, self.format, self.save_name, self.format))
        path.parent.mkdir(exist_ok=True, parents=True)
        # plt.show()
        plt.savefig(
            path,
            format=self.format,
            bbox_inches="tight",
            pad_inches=0,
            dpi=1000,
            transparent=self.transparent,
        )
        plt.close(fig)

    def plot_one_algorithm_exp(self, df, algorithm_name):
        legend_name = self.get_legend_name_by_algorithm(algorithm_name)
        color = self.get_color_by_algorithm(algorithm_name)
        df = self.filter_algorithm(df, [algorithm_name])
        step = df["step"].values
        log_10_exp = df["log_10_exp"].values
        plt.plot(step, log_10_exp, label=legend_name, color=color, linewidth=2.5)

    def get_color_by_algorithm(self, algorithm_name):
        color_dict = {
            "CFR": "#0C8599",
            "CFRPlus": "#7FC41D",
            "LinearCFR": "#F99D00",
            "DCFR": "#7D92EE",
            "DDCFR": "#A58DF9",
            "DCFRPlus": "#F16D85",
            "AutoCFR4": "#F1FA8C",
            "DCFRPlusTwo": "#F1FA8C",
            "AutoCFRS": "#282A36",
            "DCFRPlus05": "#282A36",
        }
        return color_dict[algorithm_name]

    def get_legend_name_by_algorithm(self, algorithm_name):
        legend_name_dict = {
            "CFR": "CFR",
            "CFRPlus": "CFR+",
            "LinearCFR": "Linear CFR",
            "DCFR": "DCFR",
            "DDCFR": "DDCFR",
            "DCFRPlus": "DCFR+",
            "DCFRPlusTwo": "DCFR+2",
            "AutoCFR4": "AutoCFR4",
            "AutoCFRS": "AutoCFRS",
            "DCFRPlus05": "DCFRPlus05"
        }
        return legend_name_dict[algorithm_name]

    def load_baseline(self):
        df = load_df("../baseline")
        df["log_10_exp"] = np.log(df["exp"]) / np.log(10)
        return df

    def load_algorithm(self):
        # bigleduc_df = load_df("../bigleduc")
        df_list = []
        game_name = self.game_name
        for algo_name in self.algorithm_names:
            df = load_df("../games/{}/{}_{}".format(game_name, algo_name, game_name))
            if df is not None:
                df_list.append(df)
        df = pd.concat(df_list)
        df["log_10_exp"] = np.log(df["exp"]) / np.log(10)
        return df

    def filter_algorithm(self, df, algorithm_names=None):
        if algorithm_names is None:
            algorithm_names = self.algorithm_names
        result_df = df[df.algorithm_name.isin(algorithm_names)]
        return result_df

    def filter_freq(self, df):
        result_df = df[df.step % self.print_freq == 0]
        return result_df

    def filter_iterations(self, df):
        result_df = df[df.step <= self.iterations]
        return result_df

    def filter_game(self, df, game_name_list):
        result_df = df[df.game_name.isin(game_name_list)]
        return result_df

    def set_font_size(self):
        import matplotlib.pyplot as plt

        # plt.rc('font', size=20)          # controls default text sizes
        plt.rc("axes", titlesize=13)  # fontsize of the axes title
        plt.rc("axes", labelsize=13)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=11)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=11)  # fontsize of the tick labels
        plt.rc("legend", fontsize=11)  # legend fontsize
        # plt.rc('figure', titlesize=10)  # fontsize of the figure title

    def set_yticks(self):
        tick_dict = {
            -12: "1e-12",
            -11: "1e-11",
            -10: "1e-10",
            -9: "1e-9",
            -8: "1e-8",
            -7: "1e-7",
            -6: "1e-6",
            -5: "1e-5",
            -4: "1e-4",
            -3: "0.001",
            -2: "0.01",
            -1: "0.1",
            0: "1",
            1: "10",
            2: "100",
            3: "1000",
            4: "1e4",
            5: "1e5",
        }
        tick_range_before = [i for i in self.tick_list if i in tick_dict]
        tick_range_after = [tick_dict[tick] for tick in tick_range_before]
        plt.yticks(tick_range_before, tick_range_after)

    def set_xticks(self):
        tick_list = list(range(0, self.iterations + 1, self.iterations // 4))
        tick_range_before = [i for i in tick_list]
        tick_range_after = [i for i in tick_list]
        plt.xticks(tick_range_before, tick_range_after)

    def set_lim(self):
        plt.ylim(self.ymin, self.ymax)


train_class_list = []
test_class_list = []
plot_class_list = []

def train(cls):
    train_class_list.append(cls)
    return cls


def test(cls):
    test_class_list.append(cls)
    return cls

def plot(cls):
    plot_class_list.append(cls)
    return cls

class PlotGoofspiel3DecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(3) Dec"
        self.save_name = "Goofspiel_3_dec"
        self.tick_min = -9
        self.tick_max = 0
        super().__init__("goofspiel_3_dec")


class PlotGoofspiel3DecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(3) Dec Diff"
        self.save_name = "Goofspiel_3_dec_diff"
        self.tick_min = -9
        self.tick_max = 0
        super().__init__("goofspiel_3_dec_diff")


class PlotGoofspiel3ImpDecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(3) Imp Dec Diff"
        self.save_name = "Goofspiel_3_imp_dec_diff"
        self.tick_min = -9
        self.tick_max = 0
        super().__init__("goofspiel_3_imp_dec_diff")


class PlotGoofspiel4DecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(4) Dec"
        self.save_name = "Goofspiel_4_dec"
        self.tick_min = -9
        self.tick_max = 0
        super().__init__("goofspiel_4_dec")


class PlotGoofspiel4DecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(4) Dec Diff"
        self.save_name = "Goofspiel_4_dec_diff"
        self.tick_min = -9
        self.tick_max = 0
        super().__init__("goofspiel_4_dec_diff")


class PlotGoofspiel4Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(4)"
        self.save_name = "Goofspiel_4"
        self.tick_min = -5
        self.tick_max = 0
        super().__init__("goofspiel_4")


class PlotGoofspiel4ImpDecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(4) Imp Dec Diff"
        self.save_name = "Goofspiel_4_imp_dec_diff"
        self.tick_min = -9
        self.tick_max = 0
        super().__init__("goofspiel_4_imp_dec_diff")


class PlotGoofspiel5DecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(5) Dec"
        self.save_name = "Goofspiel_5_dec"
        self.tick_min = -8
        self.tick_max = 0
        super().__init__("goofspiel_5_dec")


class PlotGoofspiel5DecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(5) Dec Diff"
        self.save_name = "Goofspiel_5_dec_diff"
        self.tick_min = -4
        self.tick_max = 1
        super().__init__("goofspiel_5_dec_diff")


class PlotGoofspiel5ImpDecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(5) Imp Dec"
        self.save_name = "Goofspiel_5_imp_dec"
        self.tick_min = -6
        self.tick_max = 1
        super().__init__("goofspiel_5_imp_dec")


class PlotGoofspiel5ImpDecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(5) Imp Dec Diff"
        self.save_name = "Goofspiel_5_imp_dec_diff"
        self.tick_min = -5
        self.tick_max = 1
        super().__init__("goofspiel_5_imp_dec_diff")


class PlotGoofspiel6DecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(6) Dec"
        self.save_name = "Goofspiel_6_dec"
        self.tick_min = -3
        self.tick_max = 1
        super().__init__("goofspiel_6_dec")


class PlotGoofspiel6DecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(6) Dec Diff"
        self.save_name = "Goofspiel_6_dec_diff"
        self.tick_min = -3
        self.tick_max = 1
        super().__init__("goofspiel_6_dec_diff")


class PlotGoofspiel6ImpDecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(6) Imp Dec"
        self.save_name = "Goofspiel_6_imp_dec"
        self.tick_min = -3
        self.tick_max = 1
        super().__init__("goofspiel_6_imp_dec")


class PlotGoofspiel6ImpDecDiffCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(6) Imp Dec Diff"
        self.save_name = "Goofspiel_6_imp_dec_diff"
        self.tick_min = -4
        self.tick_max = 1
        super().__init__("goofspiel_6_imp_dec_diff")


class PlotLiarsDice14Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Liar's Dice(1, 4)"
        self.save_name = "Liars_Dice_1_4"
        self.tick_min = -7
        self.tick_max = 0
        super().__init__("liars_dice_1n_4s")


class PlotLiarsDice16Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Liar's Dice(1, 6)"
        self.save_name = "Liars_Dice_1_6"
        self.tick_min = -5
        self.tick_max = 0
        super().__init__("liars_dice_1n_6s")


class PlotLiarsDice15Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Liar's Dice(1, 5)"
        self.save_name = "Liars_Dice_1_5"
        self.tick_min = -6
        self.tick_max = 0
        super().__init__("liars_dice_1n_5s")


@train
class PlotDCFRExampleCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "NFG-1"
        self.save_name = "NFG-1"
        self.tick_min = -3
        self.tick_max = 5
        super().__init__("rf_game", legend=True, iterations=1000, print_freq=20)

    def get_legend_name_by_algorithm(self, algorithm_name):
        legend_name_dict = {
            "CFR": "CFR",
            "CFRPlus": "CFR+",
            "LinearCFR": "Linear CFR",
            "DCFR": "DCFR",
            # "DDCFR": "DDCFR",
            # "DCFRPlus": "DCFR+",
            "DCFRPlus": "Learned CFR\nVariant",
        }
        return legend_name_dict[algorithm_name]

@train
class PlotRPS3Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "NFG-2"
        self.save_name = "NFG-2"
        self.tick_min = -4
        self.tick_max = 4
        super().__init__("trm_example", legend=False, iterations=1000, print_freq=20)


class PlotRfCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "NFG-5"
        self.save_name = "NFG-5"
        self.tick_min = -4
        self.tick_max = 4
        super().__init__("dcfr_example", legend=False, iterations=1000, print_freq=20)


@train
class PlotSmallValueCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "NFG-3"
        self.save_name = "NFG-3"
        self.tick_min = -10
        self.tick_max = 0
        super().__init__("small_value", legend=True, iterations=1000, print_freq=20)


@train
class PlotMoreActionCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "NFG-4"
        self.save_name = "NFG-4"
        self.tick_min = -5
        self.tick_max = 3
        super().__init__("more_action", legend=False, iterations=1000, print_freq=20)


@train
class PlotKuhnPokerCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Kuhn Poker"
        self.save_name = "Kuhn_Poker"
        self.tick_min = -5
        self.tick_max = 0
        super().__init__("kuhn_poker", legend=False, iterations=1000, print_freq=20)


@train
class PlotLiarsDice13Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Liar's Dice(3)"
        self.save_name = "Liars_Dice_1_3"
        self.tick_min = -8
        self.tick_max = 0
        super().__init__("liars_dice_1n_3s", legend=False, iterations=1000, print_freq=20)


@train
class PlotLiarsDice14T100Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Liar's Dice(4)"
        self.save_name = "Liars_Dice_1_4_100"
        self.tick_min = -4
        self.tick_max = 0
        super().__init__("liars_dice_1n_4s", legend=False, iterations=100, print_freq=2)


@train
class PlotGoofspiel3ImpDecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(3)"
        self.save_name = "Goofspiel_3"
        self.tick_min = -9
        self.tick_max = 0
        super().__init__("goofspiel_3_imp_dec", legend=False, iterations=1000, print_freq=20)


# @plot

@test
class PlotGoofspiel4ImpDecCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Goofspiel(4)"
        self.save_name = "Goofspiel_4"
        self.tick_min = -7
        self.tick_max = 1
        super().__init__(
            "goofspiel_4_imp_dec", legend=True, iterations=20000, print_freq=100
        )


@test
class PlotLeducPokerCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Leduc Poker"
        self.save_name = "Leduc_Poker"
        self.tick_min = -6
        self.tick_max = 1
        super().__init__("leduc_poker", legend=False, iterations=20000, print_freq=100)


@test
class PlotBigLeducPokerCompare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "Big Leduc Poker"
        self.save_name = "Big_Leduc_Poker"
        self.tick_min = -7
        self.tick_max = 1
        super().__init__("bigleduc", legend=False, iterations=20000, print_freq=100)


class PlotSubgame1Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "HUNL Subgame(1)"
        self.save_name = "Subgame1"
        self.tick_min = -4
        self.tick_max = 2
        super().__init__("subgame1", legend=True, iterations=4000, print_freq=20)


class PlotSubgame2Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "HUNL Subgame(2)"
        self.save_name = "Subgame2"
        self.tick_min = -4
        self.tick_max = 2
        super().__init__("subgame2", legend=True, iterations=4000, print_freq=20)


@test
class PlotSubgame3Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "HUNL Subgame(3)"
        self.save_name = "Subgame3"
        self.tick_min = -5
        self.tick_max = 2
        super().__init__("subgame3", legend=False, iterations=20000, print_freq=100)

@test
class PlotSubgame4Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "HUNL Subgame(4)"
        self.save_name = "Subgame4"
        self.tick_min = -6
        self.tick_max = 2
        super().__init__("subgame4", legend=False, iterations=20000, print_freq=100)

@plot
class PlotRPS3Compare(PlotAlgorithmCompare):
    def __init__(self):
        self.title_name = "test_nfg"
        self.save_name = "test_nfg"
        self.tick_min = -4
        self.tick_max = 4
        super().__init__("test_nfg", legend=True, iterations=1000, print_freq=20)

for run_class in train_class_list:
    run_class().run()

for run_class in test_class_list:
    run_class().run()

# for run_class in plot_class_list:
#     run_class().run()
