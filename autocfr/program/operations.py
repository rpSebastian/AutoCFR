import numpy as np
from autocfr.program import program_types


class Operation:
    def __init__(self, *inputs):
        self.inputs = inputs
        self.check_program_types_match_signature(inputs, self.input_program_types)
        self.cached_output_type = self.output_type_fn(
            [i.cached_output_type for i in inputs]
        )
        self.is_data_structure = False

    def check_program_types_match_signature(self, inputs, input_program_types):
        assert len(inputs) == len(input_program_types), (
            len(inputs),
            len(input_program_types),
        )
        for input_node, input_type in zip(inputs, input_program_types):
            assert program_types.equal_or_supertype(
                input_node.cached_output_type.__class__, input_type
            ), (self, "wanted", input_type, "got", input_node.output_type)

    @classmethod
    def output_type_fn(cls, input_types):
        output_type_class = cls._get_output_type_class_from_input_types(input_types)
        assert issubclass(output_type_class, program_types.Type)
        output_type = output_type_class()
        return output_type

    @classmethod
    def inputs_allowed(cls, inputs):
        for i, input in enumerate(inputs):
            input_program_type = cls.input_program_types[i]
            if input_program_type not in program_types.type_and_supertypes(
                input.get_type()
            ):
                return False
        return True

    def get_type(self):
        if hasattr(self, "var_type"):
            return self.var_type.__class__
        else:
            return self.cached_output_type.__class__


class Variable(Operation):
    def __init__(self, var_type, name):
        self.var_type = var_type
        self.output_type = var_type
        self.cached_output_type = var_type
        self.name = name
        self.is_data_structure = issubclass(
            var_type.__class__, program_types.DataStructure
        )

    def __repr__(self):
        n = self.name if self.name else ""
        return n


def get_output_type_class_V_and_V(input_types):
    if (
        input_types[0].__class__ == program_types.Vector
        or input_types[1].__class__ == program_types.Vector
    ):
        return program_types.Vector
    else:
        return program_types.Scalar


def get_output_type_class_V(input_types):
    if input_types[0].__class__ == program_types.Vector:
        return program_types.Vector
    else:
        return program_types.Scalar


class Add(Operation):
    input_program_types = (program_types.Vector, program_types.Vector)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = True

    def execute(self, input_values):
        return input_values[0] + input_values[1]

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V_and_V(input_types)


class Subtract(Operation):
    input_program_types = (program_types.Vector, program_types.Vector)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = False

    def execute(self, input_values):
        return input_values[0] - input_values[1]

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V_and_V(input_types)


class Multiply(Operation):
    input_program_types = (program_types.Vector, program_types.Vector)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = True

    def execute(self, input_values):
        return input_values[0] * input_values[1]

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V_and_V(input_types)


class Div(Operation):
    input_program_types = (program_types.Vector, program_types.Scalar)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = False

    def execute(self, input_values):
        with np.errstate(divide="ignore"):
            return input_values[0] / input_values[1]

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V(input_types)


class Exp(Operation):
    input_program_types = (program_types.Vector,)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = False

    def execute(self, input_values):
        a = input_values[0]
        with np.errstate(over="ignore"):
            return np.exp(a)

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V(input_types)


class Pow(Operation):
    input_program_types = (program_types.Vector, program_types.Scalar)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = False

    def execute(self, input_values):
        a, b = input_values
        return np.power(a, b)

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V(input_types)


class Mean(Operation):
    input_program_types = (program_types.Vector,)
    possible_output_types = [program_types.Scalar]
    commutative = False

    def execute(self, input_values):
        a = input_values[0]
        return np.mean(a)

    @classmethod
    def inputs_allowed(cls, inputs):
        return inputs[0].get_type() == program_types.Vector and super().inputs_allowed(
            inputs
        )

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return program_types.Scalar


class LT(Operation):
    input_program_types = (program_types.Vector, program_types.Scalar)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = False

    def execute(self, input_values):
        a, b = input_values
        return (a < b).astype(program_types.nptype)

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V(input_types)


class GE(Operation):
    input_program_types = (program_types.Vector, program_types.Scalar)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = False

    def execute(self, input_values):
        a, b = input_values
        return (a >= b).astype(program_types.nptype)

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V(input_types)


class Normalize(Operation):
    input_program_types = (program_types.Vector,)
    possible_output_types = [program_types.Vector]
    commutative = False

    def execute(self, input_values):
        vec = input_values[0]
        size = vec.shape[0]
        p_sum = np.sum(vec)
        if p_sum == 0:
            vec = np.ones(size, dtype=program_types.nptype) / size
        else:
            vec = vec / p_sum
        return vec

    @classmethod
    def inputs_allowed(cls, inputs):
        return inputs[0].get_type() == program_types.Vector and super().inputs_allowed(
            inputs
        )

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return program_types.Vector


class Max(Operation):
    input_program_types = (program_types.Vector, program_types.Vector)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = True

    def execute(self, input_values, profiler=None, i_episode=None):
        a, b = input_values
        return np.maximum(a, b)

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V_and_V(input_types)


class Min(Operation):
    input_program_types = (program_types.Vector, program_types.Vector)
    possible_output_types = [program_types.Vector, program_types.Scalar]
    commutative = True

    def execute(self, input_values, profiler=None, i_episode=None):
        a, b = input_values
        return np.minimum(a, b)

    @classmethod
    def _get_output_type_class_from_input_types(cls, input_types):
        return get_output_type_class_V_and_V(input_types)
