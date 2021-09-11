import attr
import numpy as np
from autocfr.program.program_types import nptype


@attr.s
class State:
    player = attr.ib()
    feature = attr.ib()
    na = attr.ib()
    reach = attr.ib()
    ins_regret = attr.ib()
    cumu_regret = attr.ib()
    strategy = attr.ib()
    cumu_strategy = attr.ib()
    average_strategy = attr.ib()

    @classmethod
    def from_history(cls, h):
        player = h.current_player()
        feature = h.information_state_string()
        na = len(h.legal_actions())
        reach = 0
        ins_regret = np.zeros([na], dtype=nptype)
        cumu_regret = np.zeros([na], dtype=nptype)
        strategy = np.ones([na], dtype=nptype) / na
        cumu_strategy = np.zeros([na], dtype=nptype)
        average_strategy = np.ones([na], dtype=nptype) / na
        return cls(
            player,
            feature,
            na,
            reach,
            ins_regret,
            cumu_regret,
            strategy,
            cumu_strategy,
            average_strategy,
        )

    def get_average_strategy(self):
        vec = self.cumu_strategy
        size = vec.shape[0]
        p_sum = np.sum(vec)
        if p_sum == 0:
            return np.ones(size, dtype=nptype) / size
        else:
            return vec / p_sum

    def execute_program(self, algorithm, iters):
        input_values_of_names = {
            "ins_regret": self.ins_regret,
            "reach_prob": nptype(self.reach),
            "iters": nptype(iters),
            "cumu_regret": self.cumu_regret,
            "strategy": self.strategy,
            "cumu_strategy": self.cumu_strategy,
        }
        output_values_of_names = algorithm.execute(input_values_of_names)
        self.ins_regret *= 0
        self.reach = 0
        self.cumu_regret = output_values_of_names["cumu_regret"]
        self.strategy = output_values_of_names["strategy"]
        self.cumu_strategy = output_values_of_names["cumu_strategy"]
        self.average_strategy = self.get_average_strategy()


class CFRSolver:
    def __init__(self, game, algorithm):
        self.game = game
        self.algorithm = algorithm
        self.states = {}
        self.np = self.game.num_players()
        self.init_states(self.game.new_initial_state())
        self.node_touched = 0
        self.iter_count = 0

    def init_states(self, h):
        if h.is_terminal():
            return
        if h.is_chance_node():
            for a, p in h.chance_outcomes():
                self.init_states(h.child(a))
            return
        feature = h.information_state_string()
        if self.states.get(feature) is None:
            self.states[feature] = State.from_history(h)
        for a in h.legal_actions():
            self.init_states(h.child(a))

    def iteration(self):
        self.iter_count += 1
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)
            for s in self.states.values():
                if s.player == i:
                    s.execute_program(self.algorithm, self.iter_count)

    def display(self):
        print(len(self.states))
        for s in self.states.values():
            if s.feature == "2pb":
                print(s)

    def average_policy(self):
        def wrap(h):
            feature = h.information_state_string()
            s = self.states[feature]
            average_policy = {}
            for index, a in enumerate(h.legal_actions()):
                average_policy[a] = s.average_strategy[index]
            return average_policy
        return wrap

    def calc_regret(self, h, traveser, my_reach, opp_reach):
        self.node_touched += 1
        if h.is_terminal():
            return h.returns()[traveser]

        if h.is_chance_node():
            v = 0
            for a, p in h.chance_outcomes():
                v += p * self.calc_regret(h.child(a), traveser, my_reach, opp_reach * p)
            return v

        # 剪枝，双方概率都为0时没有必要计算
        if opp_reach + my_reach == 0:
            return 0

        feature = h.information_state_string()
        s = self.states[feature]

        if h.current_player() != traveser:
            v = 0
            for index, a in enumerate(h.legal_actions()):
                p = s.strategy[index].item()
                v += p * self.calc_regret(h.child(a), traveser, my_reach, opp_reach * p)
            return v

        child_v = {}
        v = 0
        for index, a in enumerate(h.legal_actions()):
            p = s.strategy[index].item()
            child_v[a] = self.calc_regret(h.child(a), traveser, my_reach * p, opp_reach)
            v += p * child_v[a]

        for index, a in enumerate(h.legal_actions()):
            s.ins_regret[index] += opp_reach * (child_v[a] - v)

        # 累积当前策略的自己到达概率，用来更新平均策略。
        s.reach += my_reach
        # s.cfv_history[self.iter_count] += opp_reach * v
        # self.cumulate_policy(s, my_reach)
        return v
