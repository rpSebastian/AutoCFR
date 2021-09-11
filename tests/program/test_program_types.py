import unittest

from autocfr.program import program_types
import numpy as np


class ProgramTypesTest(unittest.TestCase):
    def test_vector(self):
        value = np.array([1, 2, 3], program_types.nptype)
        self.assertTrue(program_types.Vector.is_valid_value(value))
        self.assertTrue(program_types.Vector.value_class == type(value))

        value = np.array([np.nan, 2, 3], program_types.nptype)
        self.assertTrue(program_types.Vector.value_class == type(value))
        self.assertFalse(program_types.Vector.is_valid_value(value))

        value = np.array([np.inf, np.inf, np.inf], program_types.nptype)
        self.assertTrue(program_types.Vector.value_class == type(value))
        self.assertFalse(program_types.Vector.is_valid_value(value))

    def test_scalar(self):
        value = program_types.nptype(2)
        vec_value = np.array([1, 2, 3], program_types.nptype)
        value_32 = np.float32(2)
        self.assertTrue(program_types.Scalar.is_valid_value(value))
        self.assertTrue(program_types.Scalar.value_class == type(value))
        self.assertTrue(program_types.Scalar.is_valid_value(vec_value))
        self.assertFalse(program_types.Scalar.value_class == type(vec_value))
        self.assertFalse(program_types.Scalar.value_class == type(value_32))

    def test_constant(self):
        constant_type = program_types.Constant(2)
        constant_value = constant_type.create_empty()
        self.assertTrue(constant_value == 2)

