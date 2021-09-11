import numpy as np
from autocfr.exp import ex
from autocfr.generator.cross import cross
from autocfr.generator.early_hurdle import early_hurdle
from autocfr.generator.mutate import mutate
from autocfr.generator.program_check import program_check
from autocfr.generator.hash_encoding import hash_encoding
from autocfr.worker import VecWorker, Worker


class Generator(Worker):
    @ex.capture
    def __init__(
        self,
        index,
        check_program=True,
        programs_max_length=[12, 12, 12],
        crossover_prob=0.1,
        early_hurdle=True,
        early_hurdle_iters=100,
    ):
        super().__init__(index)
        self.check_program = check_program
        self.programs_max_length = programs_max_length
        self.crossover_prob = crossover_prob
        self.early_hurdle = early_hurdle
        self.early_hurdle_iters = early_hurdle_iters

    def run(self, task):
        """变异或交叉程序得到新智能体

        1. 对程序进行变异或交叉
        2. 检验新程序，若出现异常则重新变异/交叉
        3. 计算early hurdle分数，若出现异常则重新变异/交叉
        4. 计算程序的hash编码

        Args:
            task (dict): 变异所需信息
                parent_algorithm_A (Algorithm): 待变异/交叉的第一个程序
                parent_algorithm_B (Algorithm): 待变异/交叉的第二个程序

        Returns:
            result (dict): 结果保存字典
                state (str): succ
                eval_index （int): 评估程序的评估器编号
                agents (list): 变异/交叉得到的智能体
        """
        parent_algorithm_A = task["parent_algorithm_A"]
        parent_algorithm_B = task["parent_algorithm_B"]
        self.early_hurdle_threshold = task["early_hurdle_threshold"]
        self.check_fail = 0
        self.early_hurdle_count = 0
        agent_infos = []
        while self.early_hurdle_count < 2:
            perform_cross_prob = np.random.rand()
            if perform_cross_prob < self.crossover_prob:
                agent_infos = self.cross_algorithms(
                    parent_algorithm_A, parent_algorithm_B
                )
            else:
                agent_infos = self.mutate_algorithms(parent_algorithm_A)
            if len(agent_infos) > 0:
                break
        result = self.get_result_dict(task)
        result["agent_infos"] = agent_infos
        return result

    def cross_algorithms(self, parent_algorithm_A, parent_algorithm_B):
        algorithms = cross.cross_algorithms(parent_algorithm_A, parent_algorithm_B)
        agent_infos = []
        for algorithm in algorithms:
            agent_info = self.check_algorithm_and_collect_agent_info(algorithm)
            if agent_info:
                agent_infos.append(agent_info)
        return agent_infos

    def mutate_algorithms(self, parent_algorithm):
        algorithm = mutate.mutate_algorithm(parent_algorithm, programs_max_length=self.programs_max_length)
        agent_info = self.check_algorithm_and_collect_agent_info(algorithm)
        if agent_info:
            return [agent_info]
        else:
            return []

    def check_algorithm_and_collect_agent_info(self, algorithm):
        if self.check_program:
            result = program_check.program_check(algorithm)
            if result["status"] == "fail":
                self.check_fail += 1
                return None
        if self.early_hurdle:
            result = early_hurdle.early_hurdle_score(algorithm, early_hurdle_iters=self.early_hurdle_iters)
            if result["status"] == "fail":
                return None
        early_hurdle_score = result["score"]
        if early_hurdle_score < self.early_hurdle_threshold - 1e-6:
            self.early_hurdle_count += 1
            return None
        hash_code = hash_encoding.hash_encoding(algorithm)
        agent_info = dict(
            algorithm=algorithm,
            check_fail=self.check_fail,
            early_hurdle_score=early_hurdle_score,
            hash_code=hash_code
        )
        return agent_info


class VecGenerator(VecWorker):
    @ex.capture
    def __init__(
        self,
        num_generators,
        check_program=True,
        programs_max_length=[12, 12, 12],
        crossover_prob=0.1,
        early_hurdle=True,
        early_hurdle_iters=100,
    ):
        kwargs = {
            "check_program": check_program,
            "programs_max_length": programs_max_length,
            "crossover_prob": crossover_prob,
            "early_hurdle": early_hurdle,
            "early_hurdle_iters": early_hurdle_iters,
        }
        super().__init__(num_generators, Generator, **kwargs)

    def gen_agent_infos(self, parent_algorithm_A, parent_algorithm_B, early_hurdle_threshold):
        """并行化变异程序，将该程序加入到队列中准备变异。

        Args:
           parent_algorithm_A (Algorithm): 待交叉/变异的程序A
           parent_algorithm_B (Algorithm): 待交叉/变异的程序B
        """
        task = dict(
            parent_algorithm_A=parent_algorithm_A,
            parent_algorithm_B=parent_algorithm_B,
            early_hurdle_threshold=early_hurdle_threshold
        )
        self.add_task(task)

    def get_agent_infos(self):
        result = self.get_result()
        if result is not None:
            agent_infos = result["agent_infos"]
        else:
            agent_infos = []
        return agent_infos
