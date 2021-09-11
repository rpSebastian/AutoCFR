import numpy as np


class PostCheckError(RuntimeError):
    def __init__(self, hint):
        self.hint = hint

    def __str__(self):
        return self.hint


class ProgramExecutionError(RuntimeError):
    pass


class AlgorithmExecutionError(RuntimeError):
    def __init__(self, error_info):
        super().__init__(self)
        self.error_info = error_info

    def __str__(self):
        return str(self.error_info)


def _execute_program(
    program_operations, input_values, data_structure_values, post_check=True
):
    intermediate_values = {**input_values, **data_structure_values}
    output_values = []

    for operation in program_operations:
        input_values = [intermediate_values[i] for i in operation.inputs]
        output_value = "UNSET"
        try:
            output_value = operation.execute(input_values)
            output_values.append(output_value)
            intermediate_values[operation] = output_value
        except Exception as e:
            e.info = [
                operation,
                operation.cached_output_type,
                *[intermediate_values[i] for i in operation.inputs],
                output_value,
            ]
            raise ProgramExecutionError(e)

    for operation, output_value in zip(program_operations, output_values):
        try:
            assert operation.cached_output_type.value_class == type(output_value), (
                "wanted",
                operation.cached_output_type.value_class,
                "got",
                type(output_value),
                operation,
            )
            assert operation.cached_output_type.is_valid_value(output_value)
        except Exception as e:
            e.info = [
                operation,
                operation.cached_output_type,
                *[intermediate_values[i] for i in operation.inputs],
                output_value,
            ]
            raise ProgramExecutionError(e)

    if post_check:
        for k, v in data_structure_values.items():
            if k.name == "policy":
                policy_value = v.value()
                min_policy_value = np.min(policy_value)
                sum_policy_value = np.sum(policy_value)
                if min_policy_value < 0 or abs(sum_policy_value - 1) > 1e-3:
                    e = PostCheckError("policy value erorr")
                    e.info = (
                        "policy",
                        policy_value,
                        min_policy_value,
                        sum_policy_value,
                    )
                    raise e

            if k.name == "cum_policy":
                cum_policy_value = v.value()
                min_cum_policy_value = np.min(cum_policy_value)
                if min_cum_policy_value < 0:
                    e = PostCheckError("cum policy value erorr")
                    e.info = ("policy", cum_policy_value)
                    raise e

    return intermediate_values
