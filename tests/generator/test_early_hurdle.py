import pytest
from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.generator.early_hurdle import early_hurdle


def test_early_hurdle_score():
    cfr = load_algorithm("cfr")
    result = early_hurdle.early_hurdle_score(cfr, early_hurdle_iters=10)
    assert result["status"] == "succ"
    assert pytest.approx(result["score"], 1e-6) == 0

    dcfr = load_algorithm("dcfr")
    result = early_hurdle.early_hurdle_score(dcfr, early_hurdle_iters=10)
    assert result["status"] == "succ"
    assert pytest.approx(result["score"], 1e-6) == 1

    cfr_error = load_algorithm("cfr_error")
    result = early_hurdle.early_hurdle_score(cfr_error, early_hurdle_iters=10)
    assert result["status"] == "fail"
