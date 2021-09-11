import copy
import random
from collections import defaultdict
from queue import Queue

import numpy as np

from autocfr.program import operations, program_types
from autocfr.cfr.cfr_algorithm import CFRAlgorithm


class Mutate:
    def __init__(self):
        self.init_operation_list()

    def init_operation_list(self):
        self.operations_list = [
            operations.Add,
            operations.Subtract,
            operations.Multiply,
            operations.Div,
            operations.Exp,
            operations.Pow,
            operations.Mean,
            operations.LT,
            operations.GE,
            operations.Normalize,
            operations.Max,
            operations.Min,
        ]

    def mutate_algorithm(self, parent_algorithm, programs_max_length=[12, 12, 12]):
        mutate_program_index = np.random.randint(low=0, high=3)
        new_programs = []
        for program_index in range(3):
            parent_program = parent_algorithm.programs[program_index]
            program_max_length = programs_max_length[program_index]
            if program_index == mutate_program_index:
                mutate_program = self.mutate_program(parent_program, program_max_length)
                new_programs.append(mutate_program)
            else:
                copy_program = copy.deepcopy(parent_program)
                new_programs.append(copy_program)
        new_algorithm = CFRAlgorithm(*new_programs)
        return new_algorithm

    def mutate_program(self, old_program, max_program_length=12):
        program = copy.deepcopy(old_program)
        input_variables = program.input_variables
        data_structure_variables = program.data_structure_variables
        forward_program = program.forward_program

        variables_by_type = {}
        for var in input_variables + data_structure_variables:
            self.add_graph_node(var, var.var_type, variables_by_type)
        for operation in forward_program:
            self.add_graph_node(
                operation, operation.cached_output_type, variables_by_type
            )

        if len(forward_program) < max_program_length:
            # 增加一个算子
            while len(forward_program) < max_program_length:
                forward_program = self.add_operation(forward_program, variables_by_type)
        else:
            # 修改一个算子。随机选择一个算子，替换为一个输出类型相同，输入随机的算子。
            if random.random() < 0.95:
                forward_program = self.modify_operation(
                    forward_program, variables_by_type
                )
        program.name = None
        program.program_id = np.random.randint(100000)
        program.forward_program = forward_program
        return program

    def add_graph_node(
        self, operation: operations.Operation, var_type, variables_by_type
    ):
        var_type_class = var_type.__class__
        assert var_type_class != program_types.Type, (
            var_type_class.__base__,
            operation,
            var_type,
        )
        for t in program_types.type_and_supertypes(var_type_class):
            if t not in variables_by_type:
                variables_by_type[t] = set()
            variables_by_type[t].add(operation)

    def add_operation(self, forward_program, variables_by_type):
        # operations_list = get_operation_list()
        # new_operation = sample(operations_list)
        for constant in variables_by_type.get(program_types.Constant):
            if constant.var_type.constant_value == 0:
                zero = constant
        new_operation = operations.Add
        operation_obj = new_operation(forward_program[-1], zero)
        # while True:
        #     inputs = [
        #         self.sample(list(variables_by_type.get(input_type, [])))
        #         for input_type in new_operation.input_program_types
        #     ]
        #     if new_operation.inputs_allowed(inputs):
        #         operation_obj = new_operation(*inputs)
        #         break
        forward_program.append(operation_obj)
        self.add_graph_node(
            operation_obj, operation_obj.cached_output_type, variables_by_type
        )
        return forward_program

    def modify_operation(self, forward_program, variables_by_type):
        # 随机选择一个替换的算子
        old_operation = self.sample(
            [op for op in forward_program]
        )

        # 随机选择一个输出类型相同的算子
        valid_operations = [
            op
            for op in self.operations_list
            if old_operation.cached_output_type.__class__ in op.possible_output_types
        ]
        # print(valid_operations)
        new_operation = self.sample(valid_operations)

        # 过滤替换算子之后的算子，保证DAG.
        mask = {old_operation: 1}
        for op in forward_program:
            for op_in in op.inputs:
                if op_in in mask:
                    mask[op] = 1

        # print("old_operation", old_operation)
        # print("new_operation", new_operation)
        # 选择算子输入
        while True:
            inputs = []
            for input_type in new_operation.input_program_types:
                valid_inputs = [
                    op for op in variables_by_type.get(input_type) if op not in mask
                ]
                inputs.append(self.sample(valid_inputs))

            # 创建新的算子对象
            if new_operation.inputs_allowed(inputs):
                operation_obj = new_operation(*inputs)
                if (
                    operation_obj.cached_output_type.__class__
                    == old_operation.cached_output_type.__class__
                ):
                    # print(inputs)
                    break

        # 替换原算子
        forward_program = [
            operation_obj if op == old_operation else op for op in forward_program
        ]

        # 替换原算子的后继算子
        for op in forward_program:
            # inputs = [operation_obj if op_in == old_operation else op_in for op_in in op.inputs]
            inputs = []
            for op_in in op.inputs:
                if op_in == old_operation:
                    inputs.append(operation_obj)
                else:
                    inputs.append(op_in)
            op.inputs = inputs

        # 拓扑排序
        self.topological_sort(forward_program)

        # 维护variables_by_type
        for t, operation_set in variables_by_type.items():
            if old_operation in operation_set:
                operation_set.remove(old_operation)
        self.add_graph_node(
            operation_obj, operation_obj.cached_output_type, variables_by_type
        )
        return forward_program

    def topological_sort(self, forward_program):
        all_op = set()
        deg = defaultdict(int)
        op_output = defaultdict(list)
        for op in forward_program:
            all_op.add(op)
            for op_in in op.inputs:
                all_op.add(op_in)
                deg[op] += 1
                op_output[op_in].append(op)
        queue = Queue()
        sorted_program = []
        for op in all_op:
            if deg[op] == 0:
                queue.put(op)
        while not queue.empty():
            op = queue.get()
            if op in forward_program:
                sorted_program.append(op)
            for op_out in op_output[op]:
                deg[op_out] -= 1
                if deg[op_out] == 0:
                    queue.put(op_out)
        for i in range(len(forward_program)):
            forward_program[i] = sorted_program[i]

    def sample(self, alist):
        length = len(alist)
        index = random.randint(0, length - 1)
        return alist[index]


mutate = Mutate()
