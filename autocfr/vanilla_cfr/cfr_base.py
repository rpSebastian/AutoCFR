import collections
import attr

@attr.s
class State():
    feature = attr.ib()
    player = attr.ib()
    na = attr.ib()
    reach = attr.ib()
    opp_reach = attr.ib()
    policy = attr.ib()
    regrets = attr.ib(factory=lambda: collections.defaultdict(float))
    imm_regrets = attr.ib(factory=lambda: collections.defaultdict(float))
    cum_policy = attr.ib(factory=lambda: collections.defaultdict(float))

    cfv_history = attr.ib(factory=lambda: collections.defaultdict(float))

    @classmethod
    def from_history(cls, h):
        feature = h.information_state_string()
        player = h.current_player()
        na = len(h.legal_actions())
        reach = 0 
        opp_reach = 0
        policy = { a: 1 / na for a in h.legal_actions()}
        regrets = { a: 0 for a in h.legal_actions()}
        imm_regrets = { a: 0 for a in h.legal_actions()}
        cum_policy = {a: 0 for a in h.legal_actions()}
        return cls(feature, player, na, reach, opp_reach, policy, regrets, imm_regrets, cum_policy)

    def cumulate_policy(self, my_reach):
        for a, p in self.policy.items():
            self.cum_policy[a] += p * my_reach

    def get_average_policy(self):
        cum_sum = sum(self.cum_policy.values())
        ave_policy = {} 
        for a, cum in self.cum_policy.items():
            if cum_sum == 0:
                ave_policy[a] = 1 / self.na
            else:
                ave_policy[a] = cum / cum_sum
        return ave_policy

    def __str__(self):
        return f"fearure={self.feature}, na={self.na}, reach={self.reach}, imm_regrets={self.imm_regrets}, policy={self.policy}, regrets={self.regrets}, cum_policy={self.cum_policy}"

class CFRSolverBase():
    def __init__(self, game):
        self.game = game
        self.np = self.game.num_players()
        self.states = {}
        self.init_states(self.game.new_initial_state())
        self.node_touched = 0
        self.iter_count = 0

    def display(self):
        print(len(self.states))
        for s in self.states.values():
            if s.feature == "2pb":
                print(s)

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

    def average_policy(self):
        def wrap(h):
            feature = h.information_state_string()
            s = self.states[feature]
            return s.get_average_policy()
        return wrap
    
    def hint(self):
        pass

    def iteration(self):
        self.iter_count += 1
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)

    def calc_regret(self, h, traveser, my_reach, opp_reach):
        raise NotImplementedError