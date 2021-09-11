import numpy as np
import unittest

from autocfr.program import operations, program_types


class OperationTest(unittest.TestCase):
    def test_variable(self):
        c0 = operations.Variable(program_types.Constant(0), "0")
        self.assertTrue(c0.output_type.__class__ == program_types.Constant)

    def test_add(self):
        c0 = operations.Variable(program_types.Constant(0), "0")
        c1 = operations.Variable(program_types.Scalar(), "1")
        add = operations.Add(c0, c1)
        self.assertTrue(add.cached_output_type.__class__ == program_types.Scalar)
        self.assertTrue(operations.Add.inputs_allowed([c0, c1]))
        c0_value = c0.output_type.create_empty()
        c1_value = program_types.nptype(1)
        result = add.execute([c0_value, c1_value])
        self.assertTrue(result == 1)

    def test_substract(self):
        vec = operations.Variable(program_types.Vector(), "2")
        c1 = operations.Variable(program_types.Scalar(), "1")
        sub = operations.Subtract(vec, c1)
        self.assertTrue(operations.Add.inputs_allowed([vec, c1]))
        self.assertTrue(sub.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([1, 2, 3], dtype=program_types.nptype)
        c1_value = program_types.nptype(1)
        result = sub.execute([vec_value, c1_value])
        self.assertTrue((result == np.array([0, 1, 2])).all())

    def test_div(self):
        vec = operations.Variable(program_types.Vector(), "2")
        c1 = operations.Variable(program_types.Scalar(), "1")
        div = operations.Div(vec, c1)
        self.assertTrue(operations.Div.inputs_allowed([vec, c1]))
        self.assertFalse(operations.Div.inputs_allowed([c1, vec]))
        self.assertTrue(div.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([1, 2, 3], dtype=program_types.nptype)
        c1_value = program_types.nptype(1)
        result = div.execute([vec_value, c1_value])
        self.assertTrue((result == np.array([1, 2, 3])).all())

    def test_exp(self):
        vec = operations.Variable(program_types.Vector(), "2")
        exp = operations.Exp(vec)
        self.assertTrue(operations.Exp.inputs_allowed([vec]))
        self.assertTrue(exp.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([1, 2, 3], dtype=program_types.nptype)
        result = exp.execute([vec_value])
        self.assertTrue((result == np.exp([1, 2, 3])).all())

        scalar = operations.Variable(program_types.Scalar(), "2")
        exp = operations.Exp(scalar)
        scalar_value = program_types.nptype(1e4)
        result = exp.execute([scalar_value])
        self.assertTrue(np.isinf(result))
        self.assertFalse(exp.cached_output_type.is_valid_value(result))

    def test_pow(self):
        vec = operations.Variable(program_types.Vector(), "2")
        c1 = operations.Variable(program_types.Scalar(), "1")
        pow = operations.Pow(vec, c1)
        self.assertTrue(operations.Pow.inputs_allowed([vec, c1]))
        self.assertTrue(pow.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([1, 2, 3], dtype=program_types.nptype)
        c1_value = program_types.nptype(1)
        result = pow.execute([vec_value, c1_value])
        self.assertTrue((result == np.array([1, 2, 3])).all())

    def test_mean(self):
        vec = operations.Variable(program_types.Vector(), "2")
        mean = operations.Mean(vec)
        self.assertTrue(operations.Mean.inputs_allowed([vec]))
        self.assertTrue(mean.cached_output_type.__class__ == program_types.Scalar)
        vec_value = np.array([1, 2, 3], dtype=program_types.nptype)
        result = mean.execute([vec_value])
        self.assertTrue(result == program_types.nptype(2))
        self.assertTrue(mean.cached_output_type.is_valid_value(result))

        c1 = operations.Variable(program_types.Scalar(), "1")
        self.assertFalse(operations.Mean.inputs_allowed([c1]))

    def test_lt(self):
        vec = operations.Variable(program_types.Vector(), "2")
        c1 = operations.Variable(program_types.Scalar(), "1")
        lt = operations.LT(vec, c1)
        self.assertTrue(operations.LT.inputs_allowed([vec, c1]))
        self.assertTrue(lt.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([0, 2, 3], dtype=program_types.nptype)
        c1_value = program_types.nptype(1)
        result = lt.execute([vec_value, c1_value])
        self.assertTrue((result == np.array([1, 0, 0])).all())
        self.assertTrue(lt.cached_output_type.is_valid_value(result))

    def normalize(self):
        vec = operations.Variable(program_types.Vector(), "2")
        norm = operations.Norm(vec)
        self.assertTrue(operations.Norm.inputs_allowed([vec]))
        self.assertTrue(norm.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([1, 2, 3], dtype=program_types.nptype)
        result = norm.execute([vec_value])
        self.assertTrue((result == np.array([1 / 6, 2 / 6, 3 / 6])).all())
        self.assertTrue(norm.cached_output_type.is_valid_value(result))

        c1 = operations.Variable(program_types.Scalar(), "1")
        self.assertFalse(operations.Norm.inputs_allowed([c1]))

    def test_max(self):
        vec = operations.Variable(program_types.Vector(), "2")
        c1 = operations.Variable(program_types.Scalar(), "1")
        max = operations.Max(vec, c1)
        self.assertTrue(operations.Max.inputs_allowed([vec, c1]))
        self.assertTrue(max.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([0, 2, 3], dtype=program_types.nptype)
        c1_value = program_types.nptype(1)
        result = max.execute([vec_value, c1_value])
        self.assertTrue((result == np.array([1, 2, 3])).all())
        self.assertTrue(max.cached_output_type.is_valid_value(result))

    def test_min(self):
        vec = operations.Variable(program_types.Vector(), "2")
        c1 = operations.Variable(program_types.Scalar(), "1")
        min = operations.Min(vec, c1)
        self.assertTrue(operations.Min.inputs_allowed([vec, c1]))
        self.assertTrue(min.cached_output_type.__class__ == program_types.Vector)
        vec_value = np.array([0, 2, 3], dtype=program_types.nptype)
        c1_value = program_types.nptype(1)
        result = min.execute([vec_value, c1_value])
        self.assertTrue((result == np.array([0, 1, 1])).all())
        self.assertTrue(min.cached_output_type.is_valid_value(result))
