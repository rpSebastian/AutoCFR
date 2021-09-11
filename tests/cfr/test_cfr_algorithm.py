import copy
import numpy as np
import pytest
from autocfr.cfr.cfr_algorithm import (
    CFRErrorBuilder,
    CFRProgramConstantVariables,
    CFRProgramInputVariables,
    CFRBuilder,
    CFREmptyBuilder,
    CFRPlusBuilder,
    LinearCFRBuilder,
    DCFRBuilder,
    load_algorithm,
)
from autocfr.program.executor import AlgorithmExecutionError
from autocfr.program.program_types import nptype


class TestProgramInputVariables:
    def test_get_constant_variable_by_name(self):
        program_input_variables = CFRProgramInputVariables()
        ins_regret_variable = program_input_variables.get_input_variable_by_name(
            "ins_regret"
        )
        self.assert_variable_vector_and_name(ins_regret_variable, "ins_regret")

        with pytest.raises(Exception, match="haha not defined"):
            program_input_variables.get_input_variable_by_name("haha")

    def assert_variable_vector_and_name(self, variable, name):
        assert variable.name == name
        assert variable.var_type.value_class == np.ndarray

    def assert_variable_scalar_and_name(self, variable, name):
        assert variable.name == name
        assert variable.var_type.value_class == nptype


class TestProgramConstantVariables:
    def test_get_constant_variable_by_value(self):
        program_constants_variables = CFRProgramConstantVariables()

        zero_variable = program_constants_variables.get_constant_variable_by_value(0)
        assert zero_variable.name == "0"
        assert zero_variable.var_type.constant_value == 0

        with pytest.raises(Exception, match="0.3 not defined"):
            program_constants_variables.get_constant_variable_by_value(0.3)


class TestCFRAlgorithmBuilder:
    def test_build_cumulative_strategy_program(self):
        cfr_algorithm_builder = CFRBuilder()
        cumulative_strategy_program = cfr_algorithm_builder._build_sub_program_by_name(
            "cumulative_strategy"
        )
        assert cumulative_strategy_program.name == "cfr_cumulative_strategy"

        cumulative_regret_program = cfr_algorithm_builder._build_sub_program_by_name(
            "cumulative_regret"
        )
        assert cumulative_regret_program.name == "cfr_cumulative_regret"

        new_strategy_program = cfr_algorithm_builder._build_sub_program_by_name(
            "new_strategy"
        )
        assert new_strategy_program.name == "cfr_new_strategy"


class TestCFRAlgorithm:
    def test_initialize_programs(self):
        cfr_algorithm = self.initial_algorithm(CFRBuilder)
        assert len(cfr_algorithm.constant_values_of_variables_of_programs) == 3

    def test_cfr_execute(self):
        cfr_algorithm = self.initial_algorithm(CFRBuilder)
        input_values_of_names = self.generate_input_values_of_names()
        output_values_of_names = cfr_algorithm.execute(input_values_of_names)
        expected_output_values_of_names = {
            "ins_regret": np.array([0.1, 0.4, -0.9], dtype=nptype),
            "reach_prob": nptype(0.8),
            "iters": nptype(10),
            "cumu_regret": np.array([-0.9, 1.4, 1.1], dtype=nptype),
            "strategy": np.array([0, 0.56, 0.44], dtype=nptype),
            "cumu_strategy": np.array([1.24, 2.32, 3.24], dtype=nptype),
        }
        self.assert_dict_equal(expected_output_values_of_names, output_values_of_names)

    def test_cfr_empty_execute(self):
        cfr_algorithm = self.initial_algorithm(CFREmptyBuilder)
        input_values_of_names = self.generate_input_values_of_names()
        output_values_of_names = cfr_algorithm.execute(input_values_of_names)
        expected_output_values_of_names = {
            "ins_regret": np.array([0.1, 0.4, -0.9], dtype=nptype),
            "reach_prob": nptype(0.8),
            "iters": nptype(10),
            "cumu_regret": np.array([-1, 1, 2], dtype=nptype),
            "strategy": np.array([0.3, 0.4, 0.3], dtype=nptype),
            "cumu_strategy": np.array([1, 2, 3], dtype=nptype),
        }
        self.assert_dict_equal(expected_output_values_of_names, output_values_of_names)

    def test_cfr_plus_execute(self):
        cfr_plus_algorithm = self.initial_algorithm(CFRPlusBuilder)
        input_values_of_names = self.generate_input_values_of_names()
        output_values_of_names = cfr_plus_algorithm.execute(input_values_of_names)
        expected_output_values_of_names = {
            "ins_regret": np.array([0.1, 0.4, -0.9], dtype=nptype),
            "reach_prob": nptype(0.8),
            "iters": nptype(10),
            "cumu_regret": np.array([0, 1.4, 1.1], dtype=nptype),
            "strategy": np.array([0, 0.56, 0.44], dtype=nptype),
            "cumu_strategy": np.array([3.4, 5.2, 5.4], dtype=nptype),
        }
        self.assert_dict_equal(expected_output_values_of_names, output_values_of_names)

    def test_linear_cfr_execute(self):
        linear_cfr_algorithm = self.initial_algorithm(LinearCFRBuilder)
        input_values_of_names = self.generate_input_values_of_names()
        output_values_of_names = linear_cfr_algorithm.execute(input_values_of_names)
        expected_output_values_of_names = {
            "ins_regret": np.array([0.1, 0.4, -0.9], dtype=nptype),
            "reach_prob": nptype(0.8),
            "iters": nptype(10),
            "cumu_regret": np.array([0, 5, -7], dtype=nptype),
            "strategy": np.array([0, 1, 0], dtype=nptype),
            "cumu_strategy": np.array([3.4, 5.2, 5.4], dtype=nptype),
        }
        self.assert_dict_equal(expected_output_values_of_names, output_values_of_names)

    def test_dcfr_execute(self):
        dcfr_algorithm = self.initial_algorithm(DCFRBuilder)
        input_values_of_names = self.generate_input_values_of_names()
        output_values_of_names = dcfr_algorithm.execute(input_values_of_names)
        expected_output_values_of_names = {
            "ins_regret": np.array([0.1, 0.4, -0.9], dtype=nptype),
            "reach_prob": nptype(0.8),
            "iters": nptype(10),
            "cumu_regret": np.array([-0.4, 1.36428, 1.02857], dtype=nptype),
            "strategy": np.array([0, 0.5701, 0.4299], dtype=nptype),
            "cumu_strategy": np.array([1.05, 1.94, 2.67], dtype=nptype),
        }
        self.assert_dict_equal(expected_output_values_of_names, output_values_of_names)

    def initial_algorithm(self, algorithm_builder):
        cfr_algorithm_builder = algorithm_builder()
        cfr_algorithm = cfr_algorithm_builder.build_algorithm()
        return cfr_algorithm

    def generate_input_values_of_names(self):
        input_values_of_names = {
            "ins_regret": np.array([0.1, 0.4, -0.9], dtype=nptype),
            "reach_prob": nptype(0.8),
            "iters": nptype(10),
            "cumu_regret": np.array([-1, 1, 2], dtype=nptype),
            "strategy": np.array([0.3, 0.4, 0.3], dtype=nptype),
            "cumu_strategy": np.array([1, 2, 3], dtype=nptype),
        }
        return input_values_of_names

    def assert_dict_equal(self, input_values_of_names, output_values_of_names):
        for actual_output_value, desired_output_value in zip(
            input_values_of_names.values(), output_values_of_names.values()
        ):
            np.testing.assert_almost_equal(actual_output_value, desired_output_value, 3)

    def test_load_algorithm(self):
        dcfr_algorithm = load_algorithm("dcfr", visualize=False)
        assert len(dcfr_algorithm.constant_values_of_variables_of_programs) == 3

    def test_cfr_execute_error(self):
        cfr_algorithm = self.initial_algorithm(CFRErrorBuilder)
        input_values_of_names = self.generate_input_values_of_names()
        with pytest.raises(AlgorithmExecutionError) as e:
            cfr_algorithm.execute(input_values_of_names)

    def test_execute_not_change_input(self):
        cfr_algorithm = self.initial_algorithm(CFRBuilder)
        input_values_of_names = self.generate_input_values_of_names()
        input_values_of_names_before = copy.deepcopy(input_values_of_names)
        cfr_algorithm.execute(input_values_of_names)
        self.assert_dict_equal(input_values_of_names, input_values_of_names_before)