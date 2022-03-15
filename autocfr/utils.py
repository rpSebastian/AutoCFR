import importlib
import math
import time
from collections import defaultdict
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


class Timer:
    def __init__(self):
        self.counter = defaultdict(dict)
        self.start_time = time.time()
        self.last_time = time.time()

    def reset(self):
        self.counter = defaultdict(dict)
        self.start_time = time.time()
        self.last_time = time.time()

    def count(self, info):
        now = time.time()
        self.counter[info]["total_time"] = now - self.start_time
        self.counter[info]["pass_time"] = now - self.last_time
        self.last_time = now

    def out(self):
        for info, info_dict in self.counter.items():
            msg = "{}: {:.2f} {:.2f}".format(
                info, info_dict["pass_time"], info_dict["total_time"]
            )
            yield msg

    def print(self):
        for msg in self.out():
            print(msg)
        self.counter = defaultdict(dict)


def load_game(game_config):
    import pyspiel
    if "params" in game_config:
        params = {}
        for p, v in game_config["params"].items():
            if p == "filename":
                from pathlib import Path
                v = str(Path(__file__).absolute().parent.parent / v)
            params[p] = v
        game = pyspiel.load_game(game_config["game_name"], params)
    else:
        game = pyspiel.load_game(game_config["game_name"])
    if "transform" in game_config and game_config["transform"]:
        game = pyspiel.convert_to_turn_based(game)
    return game


def update_game_configs_by_configs(game_configs, configs):
    game_configs = [
        game_config
        for game_config in game_configs
        if game_config["long_name"] in configs.keys()
    ]
    for game_config in game_configs:
        config = configs[game_config["long_name"]]
        if "max_score" in config:
            game_config["max_score"] = config["max_score"]
        if "weight" in config:
            game_config["weight"] = config["weight"]
        if "iterations" in config:
            game_config["iterations"] = config["iterations"]
    return game_configs

def load_game_configs(mode="full"):
    game_configs = [
        {
            "long_name": "NFG-1",
            "game_name": "nfg_game",
            "params": {"filename": "nfg/NFG-1.nfg"},
            "transform": True
        },
        {
            "long_name": "NFG-2",
            "game_name": "nfg_game",
            "params": {"filename": "nfg/NFG-2.nfg"},
            "transform": True
        },
        {
            "long_name": "NFG-3",
            "game_name": "nfg_game",
            "params": {"filename": "nfg/NFG-3.nfg"},
            "transform": True
        },
        {
            "long_name": "NFG-4",
            "game_name": "nfg_game",
            "params": {"filename": "nfg/NFG-4.nfg"},
            "transform": True
        },
        {
            "long_name": "kuhn_poker",
            "game_name": "kuhn_poker",
            "params": {"players": 2},
        },
        {
            "long_name": "liars_dice_1n_3s",
            "game_name": "liars_dice",
            "params": {"numdice": 1, "dice_sides": 3},
        },
        {
            "long_name": "liars_dice_1n_4s",
            "game_name": "liars_dice",
            "params": {"numdice": 1, "dice_sides": 4},
        },
        {
            "long_name": "leduc_poker",
            "game_name": "leduc_poker",
            "params": {"players": 2},
        },
        {
            "long_name": "goofspiel_3",
            "game_name": "goofspiel",
            "params": {"num_cards": 3, "imp_info": True, "points_order": "descending"},
            "transform": True,
        },
        {
            "long_name": "goofspiel_4",
            "game_name": "goofspiel",
            "params": {"num_cards": 4, "imp_info": True, "points_order": "descending"},
            "transform": True,
        }
    ]
    if mode == "train":
        configs = {
            "NFG-1": dict(max_score=1.1, weight=1, iterations=1000),
            "NFG-2": dict(max_score=1.1, weight=1, iterations=1000),
            "NFG-3": dict(max_score=1.1, weight=1, iterations=1000),
            "NFG-4": dict(max_score=1.1, weight=1, iterations=1000),
            "kuhn_poker": dict(max_score=1.2, weight=3, iterations=1000),
            "goofspiel_3": dict(max_score=1.1, weight=1, iterations=1000),
            "liars_dice_1n_3s": dict(max_score=1.1, weight=1, iterations=1000),
            "liars_dice_1n_4s": dict(max_score=1.2, weight=5, iterations=100),
        }
        game_configs = update_game_configs_by_configs(game_configs, configs)
    elif mode == "ablation_study_train":
        configs = {
            "kuhn_poker": dict(max_score=1.2, weight=3, iterations=1000),
            "goofspiel_3": dict(max_score=1.1, weight=1, iterations=1000),
            "liars_dice_1n_3s": dict(max_score=1.1, weight=1, iterations=1000),
            "liars_dice_1n_4s": dict(max_score=1.2, weight=5, iterations=100),
        }
        game_configs = update_game_configs_by_configs(game_configs, configs)
    elif mode == "test":
        configs = {
            "goofspiel_4": dict(iterations=20000),
            "leduc_poker": dict(iterations=20000),
        }
        game_configs = update_game_configs_by_configs(game_configs, configs)
    elif mode == "full":
        configs = {
            "NFG-1": dict(iterations=1000),
            "NFG-2": dict(iterations=1000),
            "NFG-3": dict(iterations=1000),
            "NFG-4": dict(iterations=1000),
            "kuhn_poker": dict(iterations=1000),
            "goofspiel_3": dict(iterations=1000),
            "liars_dice_1n_3s": dict(iterations=1000),
            "liars_dice_1n_4s": dict(iterations=100),
            "goofspiel_4": dict(iterations=20000),
            "leduc_poker": dict(iterations=20000),
        }
        game_configs = update_game_configs_by_configs(game_configs, configs)
    else:
        raise Exception("Do not support mode {}".format(mode))
    for game_config in game_configs:
        game_config["name"] = game_config["long_name"] + "_" + str(game_config["iterations"])
    return game_configs


def load_module(name):
    print(name)
    if ":" in name:
        mod_name, attr_name = name.split(":")
    else:
        li = name.split(".")
        mod_name, attr_name = ".".join(li[:-1]), li[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def save_df(df, filename):
    csv_file = (
        Path(__file__).parent.parent
        / "models"
        / "dataframe"
        / "{}.csv".format(filename)
    )
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)


def load_df(filename):
    csv_file = (
        Path(__file__).parent.parent
        / "models"
        / "dataframe"
        / "{}.csv".format(filename)
    )
    df = pd.read_csv(csv_file)
    return df

def remove_border(path):
    img = imageio.imread(path)
    h, w, _ = img.shape  
    for u in range(0, h, 1):
        if not (img[u, :, :-1] == 255).all():
            break
    for d in range(h - 1, 0, -1):
        if not (img[d, :, :-1] == 255).all():
            break
    for l in range(0, w, 1):
        if not (img[:, l, :-1] == 255).all():
            break
    for r in range(w - 1, 0, -1):
        if not (img[:, r, :-1] == 255).all():
            break
    cropped = img[u:d+1, l:r+1, :].copy()
    imageio.imwrite(path, cropped)


def png_to_pdf(path):
    path = str(path)
    image1 = Image.open(path)
    im1 = image1.convert('RGB')
    save_path = path.replace("png", "pdf")
    im1.save(save_path)
