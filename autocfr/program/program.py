from autocfr.program import operations, program_types, executor, executor
from graphviz import Digraph


class Program:
    def __init__(
        self,
        forward_program,
        input_variables,
        data_structure_variables,
        program_id=None,
        name=None,
    ):
        self.forward_program = forward_program
        self.input_variables = input_variables
        self.data_structure_variables = data_structure_variables
        self.program_id = program_id
        self.name = name

    def execute(self, input_values, data_structure_values):
        return executor._execute_program(
            self.forward_program,
            input_values,
            data_structure_values,
        )[self.forward_program[-1]]

    def initialize_program_structures(self):
        if self.data_structure_variables is None:
            return None
        else:
            data_structure_values = {
                d: d.output_type.create_empty() for d in self.data_structure_variables
            }
            return data_structure_values

    def __str__(self):
        variables = self.input_variables + self.data_structure_variables
        var_to_name = self._variable_names(variables)
        output = []
        for operation in self.forward_program:
            c = ", ".join(var_to_name[i] for i in operation.inputs)
            output.append(
                f"{var_to_name[operation]} = {operation.__class__.__name__}({c})".ljust(
                    60
                )
                + f" # {operation.cached_output_type.__class__.__name__}".ljust(30)
            )

        f = ", ".join([var_to_name[i] for i in self.forward_program])
        output.append(f"program = Program([{f}])")
        return "\n".join(output)

    def _variable_names(self, variables):
        var_to_name = {}
        var_name_list = [chr(i) for i in range(97, 97 + 26)] + [
            chr(i) for i in range(65, 65 + 26)
        ]
        var_name_list += ["a" + chr(i) for i in range(48, 58)] + [
            "b" + chr(i) for i in range(48, 58)
        ]
        name_num = 0
        for variable in variables + self.forward_program:
            if type(variable) is operations.Variable:
                var_name = variable.name
            else:
                # var_name = chr(name_num + 97)
                var_name = var_name_list[name_num]
                name_num += 1

            var_to_name[variable] = var_name
        return var_to_name

    def visualize_as_graph(self, debug_text=None, abs_name=None):
        filename = self.name if self.name else str(self.program_id)

        def omit_variable(operation: operations.Operation):
            # TODO: Do this in a more general way
            return not (
                type(operation) == operations.Variable and operation.name == "Adam"
            )

        if abs_name:
            g = Digraph(f"{abs_name}", format="png")
        else:
            g = Digraph(f"images/program_{filename}", format="png")

        variables = []
        for op in self.forward_program:
            variables.extend(op.inputs)
            variables.append(op)
        variables = list(filter(omit_variable, variables))

        var_to_name = self._variable_names(list(set(variables)))

        def node_id(node: operations.Operation):
            return var_to_name[node]

        def node_name(node: operations.Operation):
            if isinstance(node, operations.Variable):
                if node.name is None and node.short_name is None:
                    return type(node.cached_output_type).__name__
                elif hasattr(node, "short_name") and node.short_name is not None:
                    return f"< <B> {str(node.short_name)} </B> >"
                else:
                    return f"< <B> {str(node.name)} </B> >"
            else:
                nodename = (
                    node.short_name
                    if hasattr(node, "short_name")
                    else type(node).__name__
                )
                # output_type = (
                #     type(node.cached_output_type).__name__
                #     if not hasattr(node.cached_output_type, "short_name")
                #     else node.cached_output_type.short_name
                # )
                return f"< <B>{str(nodename)}</B> >"
                # <BR/>" \
                #     + f"{output_type} >"  # ({var_to_name[node]})

        if debug_text:
            g.node("debug_text", label=str(debug_text))

        for node in set(variables):
            with g.subgraph() as c:
                peripheries = (
                    2
                    if program_types.equal_or_supertype(
                        type(node.cached_output_type), program_types.List
                    )
                    else 1
                )
                shape = "box"

                if isinstance(node, operations.Variable):
                    is_data_structure = node.name is None or node.is_data_structure
                    color = "lightgray" if is_data_structure else "lightblue"
                    bordercolor = ""
                    c.attr(
                        "node",
                        shape=shape,
                        style="filled",
                        fillcolor=color,
                        color=bordercolor,
                    )
                elif node == self.forward_program[-1]:
                    c.attr("node", shape=shape, style="filled", color="green")
                else:
                    t = type(node.cached_output_type)
                    type_colors = {
                        program_types.Scalar: "green",
                    }

                    bordercolor = "black"
                    for supertype in type_colors:
                        if program_types.equal_or_supertype(t, supertype) or (
                            program_types.equal_or_supertype(t, program_types.List)
                            and program_types.equal_or_supertype(
                                t.list_contents_type, supertype
                            )
                        ):
                            bordercolor = type_colors[supertype]

                    c.attr(
                        "node",
                        shape=shape,
                        peripheries=str(peripheries),
                        color=bordercolor,
                    )

                c.node(node_id(node), label=node_name(node))

        for node in filter(omit_variable, self.forward_program):
            for input_num, input_node in enumerate(filter(omit_variable, node.inputs)):
                with g.subgraph() as c:

                    if hasattr(node, "input_program_names"):
                        label = node.input_program_names[input_num]
                    else:
                        label = None

                    c.edge(node_id(input_node), node_id(node), label)

        g.render(view=False, cleanup=True)

    def simplify(self):
        from collections import defaultdict

        forward_program = self.forward_program
        edge = defaultdict(list)
        flag = defaultdict(int)
        for op in forward_program:
            for op_in in op.inputs:
                edge[op].append(op_in)
        from queue import Queue

        queue = Queue()
        op = forward_program[-1]
        flag[op] = 1
        queue.put(op)

        while not queue.empty():
            u = queue.get()
            for v in edge[u]:
                if not flag[v]:
                    queue.put(v)
                    flag[v] = 1
        new_forward_program = []
        for op in forward_program:
            if flag[op]:
                new_forward_program.append(op)
        self.forward_program = new_forward_program
