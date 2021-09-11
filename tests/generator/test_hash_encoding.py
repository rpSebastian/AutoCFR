from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.generator.hash_encoding import hash_encoding


def test_hash_encoding():
    cfr = load_algorithm("cfr")
    hash_code_cfr = hash_encoding.hash_encoding(cfr)
    cfr_plus = load_algorithm("cfr_plus")
    hash_code_cfr_plus = hash_encoding.hash_encoding(cfr_plus)
    assert hash_code_cfr_plus != hash_code_cfr
    cfr_same = load_algorithm("cfr")
    hash_code_cfr_same = hash_encoding.hash_encoding(cfr_same)
    assert hash_code_cfr_same == hash_code_cfr
