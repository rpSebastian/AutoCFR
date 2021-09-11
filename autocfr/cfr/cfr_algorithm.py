import copy
import pickle
from autocfr.program import operations, program_types, executor
from autocfr.program.program import Program


class CFRAlgorithm:
    def __init__(
        self,
        cumulative_strategy_program,
        cumulative_regret_program,
        new_strategy_program,
    ):
        self.cumulative_strategy_program = cumulative_strategy_program
        self.cumulative_regret_program = cumulative_regret_program
        self.new_strategy_program = new_strategy_program
        self.programs = [
            self.cumulative_strategy_program,
            self.cumulative_regret_program,
            self.new_strategy_program,
        ]
        self.initialize_programs()

    def initialize_programs(self):
        self.constant_values_of_variables_of_programs = []
        for program in self.programs:
            constant_values = program.initialize_program_structures()
            self.constant_values_of_variables_of_programs.append(constant_values)

    def execute(self, input_values_of_names):
        try:
            return self.execute_run(input_values_of_names)
        except executor.PostCheckError as e:
            raise executor.AlgorithmExecutionError(e.info)
        except executor.ProgramExecutionError as e:
            raise executor.AlgorithmExecutionError(e.args[0].info)
        except Exception as e:
            raise executor.AlgorithmExecutionError(str(e))

    def execute_run(self, input_values_of_names):
        output_values_of_names = copy.deepcopy(input_values_of_names)
        update_variable_name_of_programs = ["cumu_strategy", "cumu_regret", "strategy"]
        for program, constant_values_of_variables, update_variable_name in zip(
            self.programs,
            self.constant_values_of_variables_of_programs,
            update_variable_name_of_programs,
        ):
            input_values_of_variables = self._get_input_values_of_variables(
                output_values_of_names, program
            )
            result = program.execute(
                input_values_of_variables, constant_values_of_variables
            )
            output_values_of_names[update_variable_name] = result
        return output_values_of_names

    def _get_input_values_of_variables(self, input_values_of_names, program):
        input_values_of_variables = {
            i: input_values_of_names[i.name] for i in program.input_variables
        }
        return input_values_of_variables

    @classmethod
    def save(cls, algorithm, filename):
        with open(filename, "wb") as f:
            pickle.dump(algorithm, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            algorithm = pickle.load(f)
        return algorithm

    def __len__(self):
        length = 0
        for program in self.programs:
            length += len(program.forward_program)
        return length

    def __repr__(self):
        return "Algorithm({})".format(
            " ".join(str(len(program.forward_program)) for program in self.programs)
        )

    def visualize(self, abs_name):
        program_suffix_list = [
            "_cumulative_strategy",
            "_cumulative_regret",
            "_new_strategy",
        ]
        for program, program_suffix in zip(self.programs, program_suffix_list):
            program_abs_name = abs_name + program_suffix
            program.visualize_as_graph(abs_name=program_abs_name)


class CFRAlgorithmBuilder:
    def __init__(self, algorithm_name):
        self.program_input_variables = CFRProgramInputVariables()
        self.program_constants_variables = CFRProgramConstantVariables()
        self.algorithm_name = algorithm_name

    def build_algorithm(self):
        _cumulative_strategy_program = self._build_sub_program_by_name(
            "cumulative_strategy"
        )
        _cumulative_regret_program = self._build_sub_program_by_name(
            "cumulative_regret"
        )
        _new_strategy_program = self._build_sub_program_by_name("new_strategy")
        algorithm = CFRAlgorithm(
            _cumulative_strategy_program,
            _cumulative_regret_program,
            _new_strategy_program,
        )
        return algorithm

    def _build_sub_program_by_name(self, sub_program_name):
        full_program_name = "{}_{}".format(self.algorithm_name, sub_program_name)
        build_forward_programs_func = {
            "cumulative_strategy": self._build_cumulative_strategy_programs,
            "new_strategy": self._build_new_strategy_programs,
            "cumulative_regret": self._build_cumulative_regret_programs,
        }
        forward_programs_func = build_forward_programs_func[sub_program_name]
        forward_programs = forward_programs_func()
        program = self._compose_program(forward_programs, full_program_name)
        return program

    def _build_cumulative_strategy_programs(self):
        raise NotImplementedError()

    def _build_cumulative_regret_programs(self):
        raise NotImplementedError()

    def _build_new_strategy_programs(self):
        raise NotImplementedError()

    def _compose_program(self, forward_programs, name):
        program_inputs = self.program_input_variables.get_input_variables()
        data_structures = self.program_constants_variables.get_constant_variables()
        program = Program(forward_programs, program_inputs, data_structures, name=name)
        return program


class CFRProgramInputVariables:
    def __init__(self):
        self._input_variables = []
        self._define_input_variables()

    def _define_input_variables(self):
        self.ins_regret_variable = operations.Variable(
            program_types.Vector(), "ins_regret"
        )
        self.reach_prob_variable = operations.Variable(
            program_types.Scalar(), "reach_prob"
        )
        self.iters_variable = operations.Variable(program_types.Scalar(), "iters")
        self.cumu_regret_variable = operations.Variable(
            program_types.Vector(), "cumu_regret"
        )
        self.strategy_variable = operations.Variable(program_types.Vector(), "strategy")
        self.cumu_strategy_variable = operations.Variable(
            program_types.Vector(), "cumu_strategy"
        )
        self._input_variables.extend(
            [
                self.ins_regret_variable,
                self.reach_prob_variable,
                self.iters_variable,
                self.cumu_regret_variable,
                self.strategy_variable,
                self.cumu_strategy_variable,
            ]
        )

    def get_input_variable_by_name(self, input_name):
        for input_variable in self._input_variables:
            if input_variable.name == str(input_name):
                return input_variable
        raise Exception("{} not defined".format(input_name))

    def get_input_variables(self):
        return self._input_variables


class CFRProgramConstantVariables:
    def __init__(self):
        self._constant_variables = []
        self._define_constant_variables()

    def get_constant_variable_by_value(self, constant_value):
        for constant_variable in self._constant_variables:
            if constant_variable.name == str(constant_value):
                return constant_variable
        raise Exception("{} not defined".format(constant_value))

    def _define_constant_variables(self):
        constant_values = [0, 1, 1.5, 2, 3, 0.1, 0.01, 0.001, -0.1, -0.01, -0.001]
        for constant_value in constant_values:
            constant_variable = operations.Variable(
                program_types.Constant(constant_value), str(constant_value)
            )
            self._constant_variables.append(constant_variable)

    def get_constant_variables(self):
        return self._constant_variables


class CFREmptyBuilder(CFRAlgorithmBuilder):
    def __init__(self, name="cfr_empty"):
        super().__init__(name)

    def _build_cumulative_strategy_programs(self):
        cumu_strategy = self.program_input_variables.get_input_variable_by_name(
            "cumu_strategy"
        )
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        add_op = operations.Add(cumu_strategy, zero)
        forward_programs = [add_op]
        return forward_programs

    def _build_cumulative_regret_programs(self):
        cumu_regret = self.program_input_variables.get_input_variable_by_name(
            "cumu_regret"
        )
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        add_op = operations.Add(cumu_regret, zero)
        forward_programs = [add_op]
        return forward_programs

    def _build_new_strategy_programs(self):
        strategy = self.program_input_variables.get_input_variable_by_name("strategy")
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        add_op = operations.Add(strategy, zero)
        forward_programs = [add_op]
        return forward_programs


class CFRBuilder(CFRAlgorithmBuilder):
    def __init__(self, name="cfr"):
        super().__init__(name)

    def _build_cumulative_strategy_programs(self):
        strategy_variable = self.program_input_variables.get_input_variable_by_name(
            "strategy"
        )
        reach_variable = self.program_input_variables.get_input_variable_by_name(
            "reach_prob"
        )
        cumu_strategy = self.program_input_variables.get_input_variable_by_name(
            "cumu_strategy"
        )
        mul_op = operations.Multiply(reach_variable, strategy_variable)
        add_op = operations.Add(mul_op, cumu_strategy)
        forward_programs = [mul_op, add_op]
        return forward_programs

    def _build_cumulative_regret_programs(self):
        ins_regret = self.program_input_variables.get_input_variable_by_name(
            "ins_regret"
        )
        cumu_regret = self.program_input_variables.get_input_variable_by_name(
            "cumu_regret"
        )
        add_op = operations.Add(cumu_regret, ins_regret)
        forward_programs = [add_op]
        return forward_programs

    def _build_new_strategy_programs(self):
        regret = self.program_input_variables.get_input_variable_by_name("cumu_regret")
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        max_op = operations.Max(zero, regret)
        norm_op = operations.Normalize(max_op)
        forward_programs = [max_op, norm_op]
        return forward_programs


class CFRPlusBuilder(CFRBuilder):
    def __init__(self, name="cfr_plus"):
        super().__init__(name)

    def _build_cumulative_regret_programs(self):
        ins_regret = self.program_input_variables.get_input_variable_by_name(
            "ins_regret"
        )
        cumu_regret = self.program_input_variables.get_input_variable_by_name(
            "cumu_regret"
        )
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        add_op = operations.Add(cumu_regret, ins_regret)
        max_op = operations.Max(add_op, zero)
        forward_programs = [add_op, max_op]
        return forward_programs

    def _build_cumulative_strategy_programs(self):
        strategy_variable = self.program_input_variables.get_input_variable_by_name(
            "strategy"
        )
        reach_variable = self.program_input_variables.get_input_variable_by_name(
            "reach_prob"
        )
        cumu_strategy = self.program_input_variables.get_input_variable_by_name(
            "cumu_strategy"
        )
        iters = self.program_input_variables.get_input_variable_by_name("iters")
        mul_op = operations.Multiply(reach_variable, strategy_variable)
        mul_op_2 = operations.Multiply(mul_op, iters)
        add_op = operations.Add(cumu_strategy, mul_op_2)
        forward_programs = [mul_op, mul_op_2, add_op]
        return forward_programs


class CFRSameBuilder(CFRBuilder):
    def __init__(self, name="cfr_same"):
        super().__init__(name)

    def _build_cumulative_regret_programs(self):
        ins_regret = self.program_input_variables.get_input_variable_by_name(
            "ins_regret"
        )
        cumu_regret = self.program_input_variables.get_input_variable_by_name(
            "cumu_regret"
        )
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        add_op = operations.Add(cumu_regret, ins_regret)
        add_op_2 = operations.Add(add_op, zero)
        forward_programs = [add_op, add_op_2]
        return forward_programs


class CFRErrorBuilder(CFRBuilder):
    def __init__(self, name="cfr_error"):
        super().__init__(name)

    def _build_cumulative_regret_programs(self):
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        one = self.program_constants_variables.get_constant_variable_by_value(1)
        div_op = operations.Div(one, zero)
        forward_programs = [div_op]
        return forward_programs


class LinearCFRBuilder(CFRBuilder):
    def __init__(self, name="linear_cfr"):
        super().__init__(name)

    def _build_cumulative_regret_programs(self):
        ins_regret = self.program_input_variables.get_input_variable_by_name(
            "ins_regret"
        )
        cumu_regret = self.program_input_variables.get_input_variable_by_name(
            "cumu_regret"
        )
        iters = self.program_input_variables.get_input_variable_by_name("iters")
        mul_op = operations.Multiply(ins_regret, iters)
        add_op = operations.Add(cumu_regret, mul_op)
        forward_programs = [mul_op, add_op]
        return forward_programs

    def _build_cumulative_strategy_programs(self):
        strategy_variable = self.program_input_variables.get_input_variable_by_name(
            "strategy"
        )
        reach_variable = self.program_input_variables.get_input_variable_by_name(
            "reach_prob"
        )
        cumu_strategy = self.program_input_variables.get_input_variable_by_name(
            "cumu_strategy"
        )
        iters = self.program_input_variables.get_input_variable_by_name("iters")
        mul_op = operations.Multiply(reach_variable, strategy_variable)
        mul_op_2 = operations.Multiply(mul_op, iters)
        add_op = operations.Add(cumu_strategy, mul_op_2)
        forward_programs = [mul_op, mul_op_2, add_op]
        return forward_programs


class DCFRBuilder(CFRBuilder):
    def __init__(self, name="dcfr"):
        super().__init__(name)

    def _build_cumulative_strategy_programs(self):
        strategy_variable = self.program_input_variables.get_input_variable_by_name(
            "strategy"
        )
        reach_variable = self.program_input_variables.get_input_variable_by_name(
            "reach_prob"
        )
        cumu_strategy = self.program_input_variables.get_input_variable_by_name(
            "cumu_strategy"
        )
        iters = self.program_input_variables.get_input_variable_by_name("iters")
        one = self.program_constants_variables.get_constant_variable_by_value(1)
        two = self.program_constants_variables.get_constant_variable_by_value(2)
        sub_op = operations.Subtract(iters, one)
        div_op = operations.Div(sub_op, iters)
        pow_op = operations.Pow(div_op, two)
        mul_op = operations.Multiply(cumu_strategy, pow_op)
        mul_op_2 = operations.Multiply(reach_variable, strategy_variable)
        add_op = operations.Add(mul_op, mul_op_2)
        forward_programs = [sub_op, div_op, pow_op, mul_op, mul_op_2, add_op]
        return forward_programs

    def _build_cumulative_regret_programs(self):
        ins_regret = self.program_input_variables.get_input_variable_by_name(
            "ins_regret"
        )
        cumu_regret = self.program_input_variables.get_input_variable_by_name(
            "cumu_regret"
        )
        iters = self.program_input_variables.get_input_variable_by_name("iters")
        one = self.program_constants_variables.get_constant_variable_by_value(1)
        zero = self.program_constants_variables.get_constant_variable_by_value(0)
        c1d5 = self.program_constants_variables.get_constant_variable_by_value(1.5)
        sub_op = operations.Subtract(iters, one)
        pow_op = operations.Pow(sub_op, c1d5)
        add_op = operations.Add(pow_op, one)
        div_op = operations.Div(pow_op, add_op)
        pow_op_2 = operations.Pow(sub_op, zero)
        add_op_2 = operations.Add(pow_op_2, one)
        div_op_2 = operations.Div(pow_op_2, add_op_2)
        little = operations.LT(cumu_regret, zero)
        bigger = operations.GE(cumu_regret, zero)
        part_1 = operations.Multiply(bigger, div_op)
        part_2 = operations.Multiply(little, div_op_2)
        add_op_3 = operations.Add(part_1, part_2)
        mul_op_2 = operations.Multiply(cumu_regret, add_op_3)
        add_op_4 = operations.Add(mul_op_2, ins_regret)
        forward_programs = [
            sub_op,
            pow_op,
            add_op,
            div_op,
            pow_op_2,
            add_op_2,
            div_op_2,
            little,
            bigger,
            part_1,
            part_2,
            add_op_3,
            mul_op_2,
            add_op_4,
        ]
        return forward_programs


def load_algorithm(algorithm_name, visualize=False):
    build_dict = {
        "cfr": CFRBuilder,
        "linear_cfr": LinearCFRBuilder,
        "cfr_plus": CFRPlusBuilder,
        "dcfr": DCFRBuilder,
        "cfr_error": CFRErrorBuilder,
        "cfr_same": CFRSameBuilder,
        "empty": CFREmptyBuilder,
    }
    if algorithm_name not in build_dict:
        raise Exception("Do not support algorithm: {}".format(algorithm_name))
    algorithm_builder = build_dict[algorithm_name]()
    algorithm = algorithm_builder.build_algorithm()
    if visualize:
        algorithm.visualize()
    return algorithm
