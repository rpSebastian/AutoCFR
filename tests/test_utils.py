import pytest
from autocfr.utils import load_module


def test_load_module():
    with pytest.raises(ModuleNotFoundError):
        load_module("autocfr.vanilla_acfr.cfr:CFRSolver")

    with pytest.raises(AttributeError):
        load_module("autocfr.vanilla_cfr.cfr:ACFRSolver")


