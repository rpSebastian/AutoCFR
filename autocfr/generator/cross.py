import copy

import numpy as np

from autocfr.cfr.cfr_algorithm import CFRAlgorithm


class Cross:
    def cross_algorithms(self, parent_algorithm_A, parent_algorithm_B):
        cross_program_index = np.random.randint(low=0, high=3)
        new_algorithms = [
            self.cross_algorithm_A_from_B(
                parent_algorithm_A, parent_algorithm_B, cross_program_index
            ),
            self.cross_algorithm_A_from_B(
                parent_algorithm_B, parent_algorithm_A, cross_program_index
            ),
        ]
        return new_algorithms

    def cross_algorithm_A_from_B(
        self, parent_algorithm_A, parent_algorithm_B, cross_program_index
    ):
        new_programs = []
        for program_index in range(3):
            parent_program_A = parent_algorithm_A.programs[program_index]
            parent_program_B = parent_algorithm_B.programs[program_index]
            if program_index == cross_program_index:
                copy_program_B = copy.deepcopy(parent_program_B)
                new_programs.append(copy_program_B)
            else:
                copy_program_A = copy.deepcopy(parent_program_A)
                new_programs.append(copy_program_A)
        new_algorithm = CFRAlgorithm(*new_programs)
        return new_algorithm


cross = Cross()
