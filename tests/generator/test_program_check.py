from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.generator.program_check import program_check


def test_program_check():
    cfr_error = load_algorithm("cfr_error")
    result = program_check.program_check(cfr_error)
    assert result["status"] == "fail"

    cfr = load_algorithm("cfr")
    result = program_check.program_check(cfr)
    assert result["status"] == "succ"
