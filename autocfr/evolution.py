import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from autocfr.cfr.cfr_algorithm import CFRAlgorithm, load_algorithm
from autocfr.evaluator import GroupVecEvaluator, VecEvaluator
from autocfr.exp import ex
from autocfr.generator.generator import VecGenerator
from autocfr.population import Agent, AgentCounter, Population
from autocfr.standing import standing
from autocfr.generator.mutate import mutate


class Evolution:
    @ex.capture
    def __init__(self, population_size, tournament_size):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.scores = set()
        self.agent_counter = AgentCounter()

    def initial(self):
        self.initial_logger()
        self.logger_info()
        self.init_game_config()
        self.init_population()
        self.init_tensorboard()
        self.init_remote_actor()

    @ex.capture
    def start_evolve(self):
        """循环执行演化算法。
        0. 评估器队列已满，阻塞运行
        1. 从种群中采样tournament_size个个体。
        2. 对采样的个体进行竞标赛，选取评分最高的个体。
        3. 对评分最高的个体进行变异，得到新个体。
        4. 将新个体加入评估队列中
        5. 每产生save_freq个体，保存种群

        Args:
            save_freq (int): 种群保存间隔，每产生save_freq个个体保存一次种群。
            num_evaluators (int): 并行评估模型的进程数
        """
        while True:
            time.sleep(0.01)

            if self.generator_free():
                parent_A = self.population.compete()
                parent_B = self.population.compete()
                self.join_generating_queue(parent_A, parent_B)

            agents = self.fetch_generating_result()

            for agent in agents:
                agent_evaluted = self.population.check_func_equal_previous_agents(agent)
                if agent_evaluted:
                    self.copy_score_and_join_popu_direct(agent)
                else:
                    if self.evaluator_free():
                        self.join_evaluating_queue(agent)
                    else:
                        self.drop_agent()

            self.fetch_evaluating_result()

            if agents:
                self.logger_status()

    def generator_free(self):
        generator_state = self.generator.state()
        return generator_state["waiting"] <= self.num_generators

    def evaluator_free(self):
        evaluator_state = self.evaluator.state()
        return evaluator_state["waiting"] <= self.num_evaluators

    def drop_agent(self):
        self.agent_counter.generaing_to_drop()

    def join_generating_queue(self, parent_A, parent_B):
        self.generator.gen_agent_infos(
            parent_A.algorithm, parent_B.algorithm, self.population.early_hurdle_threshold
        )

    def fetch_generating_result(self):
        agent_infos = self.generator.get_agent_infos()
        agents = []
        for agent_info in agent_infos:
            agent = Agent(
                agent_info["algorithm"],
                hash_code=agent_info["hash_code"],
                early_hurdle_score=agent_info["early_hurdle_score"],
            )
            agents.append(agent)
            self.agent_counter.cum_generating()
        if len(agents) == 0:
            return []
        return agents

    def copy_score_and_join_popu_direct(self, agent):
        func_equal_agent = self.population.get_func_equal_agent(agent)
        assert func_equal_agent.hash_code == agent.hash_code
        assert (
            abs(func_equal_agent.early_hurdle_score - agent.early_hurdle_score) < 1e-6
        )
        agent.copy_score(func_equal_agent)
        self.population.add_agent(agent)
        self.agent_counter.generating_to_func_equv()

    def join_evaluating_queue(self, agent):
        self.evaluator.eval_algorithm_parallel(
            agent.index, agent.algorithm, self.game_configs
        )
        self.agent_counter.generaing_to_evaluating()

    @ex.capture
    def fetch_evaluating_result(self):
        """从Evaluator中获取已评估完成的智能体。

        若智能体的程序出现错误，设置该智能体的分数为-10，打印程序异常原因。
        若智能体的程序正常运行，将智能体的利用率转换为分数，将智能体加入种群。
        """

        result = self.evaluator.get_evaluating_result()
        if result is None:
            return
        if result["status"] == "succ":
            self.process_evaluate_succ_result(result["results"])
        if result["status"] == "fail":
            self.process_evaluate_fail_result(result["results"][0])

    @ex.capture
    def process_evaluate_succ_result(self, results, agent_save_freq):
        for result in results:
            agent_index = result["agent_index"]
            agent = Agent.get_agent(agent_index)
            self.set_agent_score_from_result(agent, result)
        self.agent_counter.evaluating_to_succ()
        self.population.add_agent(agent)
        score = agent.ave_score
        max_score = self.population.max_score
        self.writer.add_scalar("max_score_by_succ", max_score, self.agent_counter.succ)
        self.file_logger.log_scalar("max_score_by_succ", max_score, self.agent_counter.succ)
        self.writer.add_scalar("max_score_by_eval", max_score, self.agent_counter.succ + self.agent_counter.fail)
        self.file_logger.log_scalar("max_score_by_eval", max_score, self.agent_counter.succ + self.agent_counter.fail)
        if self.agent_counter.succ % agent_save_freq == 0:
            agent.save()
        round_score = round(agent.ave_score, 3)
        if round_score not in self.scores:
            self.writer.add_scalar("new_ave_score", score, agent_index)
            self.file_logger.log_scalar("new_ave_score", score, agent_index)
            self.scores.add(round_score)
            agent.save(prefix="valid")

    def process_evaluate_fail_result(self, result):
        agent_index = result["agent_index"]
        game_config = result["game_config"]
        info = result["info"]
        traceback = result["traceback"]
        agent = Agent.get_agent(agent_index)
        self.agent_counter.evaluating_to_fail()
        agent.save(prefix="fail")
        max_score = self.population.max_score
        self.writer.add_scalar("max_score_by_eval", max_score, self.agent_counter.succ + self.agent_counter.fail)
        self.file_logger.log_scalar("max_score_by_eval", max_score, self.agent_counter.succ + self.agent_counter.fail)
        self.logger.warning(
            "Program {} evaluation exception, program name: {}, error info: {}\n{}".format(
                agent_index, game_config["name"], info, str(traceback)
            )
        )

    @ex.capture
    def logger_status(self, save_freq, print_freq):
        total_index = Agent.total_index
        if total_index % save_freq == 0:
            self.save_population()
        if total_index % print_freq == 0:
            self.logger.info(
                "{} evaluator: ".format(total_index) + self.evaluator.info()
            )
            self.logger.info(
                "{} generator: ".format(total_index) + self.generator.info()
            )
            self.logger.info(
                "{} Agent Counter: ".format(total_index) + self.agent_counter.info()
            )
        if total_index % print_freq == 0:
            max_score = self.population.max_score
            self.writer.add_scalar("max_score", max_score, Agent.total_index)
            self.file_logger.log_scalar("max_score", max_score, Agent.total_index)
            min_score = self.population.get_score(0)
            score_25 = self.population.get_score(25)
            score_50 = self.population.get_score(50)
            score_75 = self.population.get_score(75)
            self.writer.add_scalar("min_score", min_score, Agent.total_index)
            self.writer.add_scalar("score_25", score_25, Agent.total_index)
            self.writer.add_scalar("score_50", score_50, Agent.total_index)
            self.writer.add_scalar("score_75", score_75, Agent.total_index)

    @ex.capture
    def initial_logger(self, _log, _run):
        self.logger = _log
        self.file_logger = _run

    def logger_info(self):
        operations_list = mutate.operations_list
        names = [operation.__name__ for operation in operations_list]
        self.logger.info("  ".join(names))
        empty = load_algorithm("empty")
        self.logger.info(
            "  ".join(
                list(
                    map(str, empty.constant_values_of_variables_of_programs[0].values())
                )
            )
        )

    @ex.capture
    def init_game_config(self, game_configs):
        """初始化待评估的游戏配置。"""
        self.game_configs = game_configs
        for game_config in game_configs:
            self.logger.info(game_config)

    @ex.capture
    def init_population(self, init_population_file=None, init_algorithms_file=None):
        """初始化种群。 初始化空程序即为从零开始搜素。

        若指定了种群文件则从种群文件中读取所有个体。
        若指定了初始化程序文件，读取并评估每个程序后加入种群。

        Args:
            init_population_file (str, optional): 初始化程序文件。. Defaults to None.
            init_programs_file (str, optional): 初始化种群文件. Defaults to None.
        """
        if init_population_file is not None:
            self.init_population_from_file(init_population_file)
        elif init_algorithms_file is not None:
            self.init_population_from_algorithms(init_algorithms_file)
        else:
            raise Exception(
                "Must specify init_population_file or init_algorithms_file \n"
                "Initializing an empty algorithm equals to search from scrach"
            )

    def init_population_from_file(self, init_population_file):
        self.logger.info("start reading population.")
        self.population = Population.init_from_file(init_population_file)
        self.logger.info("finish reading population.")

    def init_population_from_algorithms(self, init_algorithms_file):
        self.logger.info("start reading algorithms from file and evaluating agents.")
        # init agents from algorithms
        agents = []
        for init_algorithm_file in init_algorithms_file:
            algorithm = CFRAlgorithm.load(init_algorithm_file)
            agent = Agent(algorithm)
            agents.append(agent)

        self.evaluate_agents_parallel(agents)

        # add agent to population
        self.population = Population(self.population_size, self.tournament_size)
        for agent in agents:
            self.population.add_agent(agent)
            self.logger.info(
                "Algorithm length: {} ave_score: {}".format(
                    len(agent.algorithm), agent.ave_score
                )
            )
        self.save_population()
        self.logger.info("Finish reading and evaluating agents.")

    def evaluate_agents_parallel(self, agents):
        # generate tasks
        evaluator = VecEvaluator(len(agents) * len(self.game_configs))
        tasks = []
        for game_config in self.game_configs:
            for agent in agents:
                task = dict(
                    agent_index=agent.index,
                    algorithm=agent.algorithm,
                    game_config=game_config,
                )
                tasks.append(task)

        # execute tasks, s.t. evaluate each agent in each game_config
        results = evaluator.execute_tasks(tasks)

        # set scores
        for result in results:
            agent_index = result["agent_index"]
            agent = Agent.get_agent(agent_index)
            self.set_agent_score_from_result(agent, result, verbose=True)

        del evaluator

    @ex.capture
    def init_tensorboard(self, _run):
        run_id = _run._id if _run is not None else "test"
        dir_name = Path(__file__).parent.parent / "logs" / str(run_id)
        self.writer = SummaryWriter(dir_name)

    @ex.capture
    def init_remote_actor(self, num_evaluators, num_generators):
        self.evaluator = GroupVecEvaluator(num_evaluators)
        self.generator = VecGenerator(num_generators)
        self.num_generators = num_generators
        self.num_evaluators = num_evaluators

    @ex.capture
    def save_population(self, _run=None):
        run_id = _run._id if _run is not None else "test"
        file = (
            Path(__file__).parent.parent
            / "logs"
            / str(run_id)
            / "models"
            / "models.pkl"
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        Population.save(self.population, str(file))

    def set_agent_score_from_result(self, agent, result, verbose=False):
        """读取评估结果，设置评估智能体的分数。

        Args:
            agent (Agent): 智能体。
            result (dict): 结果保存字典[正常运行]
                state: succ
                eval_index: 评估程序的评估器编号
                agent_index: 评估程序对应的智能体编号
                game_config: 评估游戏配置
                conv: 评估程序的利用率
            result (dict): 结果保存字典[异常]
                state: fail
                eval_index: 评估程序的评估器编号
                agent_index: 评估程序对应的智能体编号
                game_config: 评估游戏配置
                error: 异常对象
                info: 异常信息
            verbose (str): 是否打印评估信息
        """
        conv = result["conv"]
        game_config = result["game_config"]
        game_name = game_config["name"]
        score = standing.score(conv, game_config)
        agent.set_score(game_name, score, game_config["weight"])
        if verbose:
            self.logger.info(
                "Agent {} obtain score {} in Game {}".format(
                    agent.index, game_config["name"], score
                )
            )
