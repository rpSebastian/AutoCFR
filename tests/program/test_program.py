from autocfr.program import operations, program_types
from autocfr.program.program import Program


def test_program():
    # 1 + 2 + 3
    c3 = operations.Variable(program_types.Constant(3), "3")
    c1 = operations.Variable(program_types.Scalar(), "1")
    c2 = operations.Variable(program_types.Scalar(), "2")
    add = operations.Add(c3, c1)
    add2 = operations.Add(add, c2)
    add_program = Program([add, add2], [c1, c2], [c3], name="test")
    data_structure_values = add_program.initialize_program_structures()
    input_values = {
        c1: program_types.nptype(1),
        c2: program_types.nptype(2),
    }
    result = add_program.execute(input_values, data_structure_values)
    assert result == 6
