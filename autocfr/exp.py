import sys
from pathlib import Path
import logging

from sacred import Experiment

ex = Experiment("default")
logger = logging.getLogger("mylogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname).1s] %(filename)s:%(lineno)d - %(message)s ",
    "%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

ex.logger = logger

from sacred.observers import FileStorageObserver

ex.observers.append(FileStorageObserver("logs"))

"""
In each of the files in the package I do

from exp import ex

and the decorators and config variable passing seems to work. I can change the name of the experiment on the command line with --name:

$> python main.py --name newname
"""
