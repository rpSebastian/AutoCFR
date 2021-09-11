import numpy as np
from autocfr.vanilla_cfr.cfr_base import CFRSolverBase
from loguru import logger


class CFRSolver(CFRSolverBase):
    def iteration(self):
        self.iter_count += 1
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)
            # self.display()
            for s in self.states.values():
                if s.player == i:

                    # 根据当前策略计算当前轮的遗憾值和到达概率
                    self.cumulate_regret(s)

                    # 累积当前策略到平均策略中。
                    self.cumulate_policy(s)

                    # 根据当前的总遗憾值计算新的策略。
                    self.update_current_policy(s)
            # self.display()

        if self.iter_count % 1 == 0:
            self.hint()
        # for s in self.states.values():

    def hint(self):
        if False:
            for s in self.states.values():
                if len(s.imm_regrets) == 3:
                    T = float(self.iter_count)
                    print(
                        T,
                        "regrets",
                        list(s.regrets.values()),
                        "policy",
                        list(s.policy.values()),
                        "ave_policy",
                        list(s.get_average_policy().values()),
                    )

    def calc_regret(self, h, traveser, my_reach, opp_reach):
        # print(my_reach, opp_reach)
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
            for a, p in s.policy.items():
                v += p * self.calc_regret(h.child(a), traveser, my_reach, opp_reach * p)
            return v

        child_v = {}
        v = 0
        for a, p in s.policy.items():
            child_v[a] = self.calc_regret(h.child(a), traveser, my_reach * p, opp_reach)
            v += p * child_v[a]

        for a in h.legal_actions():
            s.imm_regrets[a] += opp_reach * (child_v[a] - v)

        # 累积当前策略的自己到达概率，用来更新平均策略。
        s.reach += my_reach
        s.opp_reach = max(opp_reach, s.opp_reach)
        s.cfv_history[self.iter_count] += opp_reach * v
        # self.cumulate_policy(s, my_reach)
        return v

    def cumulate_regret(self, s):
        for a in s.regrets.keys():
            s.regrets[a] += s.imm_regrets[a]
            s.imm_regrets[a] = 0

    def update_current_policy(self, s):
        regret_sum = 0
        for regret in s.regrets.values():
            regret_sum += max(0, regret)
        for a, regret in s.regrets.items():
            if regret_sum == 0:
                s.policy[a] = 1 / s.na
            else:
                s.policy[a] = max(0, regret) / regret_sum

    def cumulate_policy(self, s):
        for a, p in s.policy.items():
            s.cum_policy[a] += p * s.reach
        s.reach = 0


class CFRPlusSolver(CFRSolver):
    def cumulate_regret(self, s):
        for a in s.regrets.keys():
            s.regrets[a] = max(s.regrets[a] + s.imm_regrets[a], 0)
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = self.iter_count
        for a, p in s.policy.items():
            s.cum_policy[a] += T * p * s.reach
        s.reach = 0


class LinearCFRSolver(CFRSolver):
    def cumulate_regret(self, s):
        T = self.iter_count
        for a in s.regrets.keys():
            s.regrets[a] = s.regrets[a] + T * s.imm_regrets[a]
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = self.iter_count
        for a, p in s.policy.items():
            s.cum_policy[a] += T * p * s.reach
        s.reach = 0


class DCFRSolver(CFRSolver):
    # alpha = 1
    # beta = 1
    # gamma = 1
    alpha = 1.5
    beta = 0
    gamma = 2

    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            if s.regrets[a] > 0:
                s.regrets[a] = (
                    s.regrets[a]
                    * (np.power(T - 1, self.alpha) / (np.power(T - 1, self.alpha) + 1))
                    + s.imm_regrets[a]
                )
            else:
                s.regrets[a] = (
                    s.regrets[a]
                    * (np.power(T - 1, self.beta) / (np.power(T - 1, self.beta) + 1))
                    + s.imm_regrets[a]
                )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * np.power((T - 1) / T, self.gamma) + s.reach * p
            )
        s.reach = 0


class BetaDCFRSolver(DCFRSolver):
    Beta = 1

    def update_current_policy(self, s):
        for a, regret in s.regrets.items():
            s.policy[a] = max(0, s.regrets[a])
            s.policy[a] = np.power(s.policy[a], self.Beta)

        policy_sum = 0
        for policy in s.policy.values():
            policy_sum += max(0, policy)
        for a, policy in s.policy.items():
            if policy_sum == 0:
                s.policy[a] = 1 / s.na
            else:
                s.policy[a] = max(0, policy) / policy_sum


class Beta0d9DCFRSolver(BetaDCFRSolver):
    Beta = 0.8


class Beta0d8DCFRSolver(BetaDCFRSolver):
    Beta = 0.9


class Beta1d1DCFRSolver(BetaDCFRSolver):
    Beta = 1.1


class Beta1d2DCFRSolver(BetaDCFRSolver):
    Beta = 1.2


class ECFRSolver(CFRSolver):
    def cumulate_regret(self, s):
        # beta = -0.0001
        mean_regret = np.mean([r for r in s.imm_regrets.values()])
        s.L1 = {a: r - mean_regret for a, r in s.imm_regrets.items()}
        for a in s.regrets.keys():

            beta = -np.power(s.imm_regrets[a], 2)
            # self.beta = 1 / np.power(self.iter_count, 3)
            # self.beta = np.power(s.imm_regrets[a], 5)
            # self.beta = np.power(self.iter_count, 2)
            if s.imm_regrets[a] > 0:
                s.regrets[a] = s.regrets[a] + np.exp(s.L1[a]) * s.imm_regrets[a]
            else:
                # print(self.beta)
                s.regrets[a] = s.regrets[a] + np.exp(s.L1[a]) * beta
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = self.iter_count
        for a, p in s.policy.items():
            s.cum_policy[a] = s.cum_policy[a] + np.exp(s.L1[a]) * s.reach * p
        s.reach = 0

    def update_current_policy(self, s):
        regret_sum = 0
        for a, regret in s.regrets.items():
            regret_sum += max(0, regret) * np.exp(s.L1[a])
        for a, regret in s.regrets.items():
            if regret_sum == 0:
                s.policy[a] = 1 / s.na
            else:
                s.policy[a] = max(0, regret) * np.exp(s.L1[a]) / regret_sum


class Program108604Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = (
                min(
                    max(
                        max(s.regrets[a], int(s.regrets[a] >= 0)),
                        min(s.imm_regrets[a], (T - 1) / T),
                    )
                    * s.regrets[a]
                    * np.power(T - 1, 1.5)
                    / (np.power(T - 1, 1.5) + 1.5),
                    1,
                )
                + s.imm_regrets[a]
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T - 1, 3) * p
            )
        s.reach = 0


class Test(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = (
                min(
                    max(
                        max(s.regrets[a], int(s.regrets[a] >= 0)),
                        min(s.imm_regrets[a], (T - 1) / T),
                    )
                    * s.regrets[a]
                    * np.power(T - 1, 1.5)
                    / (np.power(T - 1, 1.5) + 1.5),
                    1,
                )
                + s.imm_regrets[a]
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T - 1, 3) * p
            )
        s.reach = 0


class P1(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = max(
                min(
                    s.regrets[a] * np.power(T - 1, 1.5) / (np.power(T - 1, 1.5) + 1.5),
                    1,
                )
                + s.imm_regrets[a],
                0,
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T - 1, 3) * p
            )
        s.reach = 0


class P2(CFRSolver):
    def cumulate_regret(self, s):
        T = self.iter_count
        for a in s.regrets.keys():
            s.regrets[a] = max(min(s.regrets[a], T) + T * s.imm_regrets[a], 0)
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T - 1, 3) * p
            )
        s.reach = 0


class DuelingCFRSolver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = max(
                min(
                    s.regrets[a] * np.power(T - 1, 1.5) / (np.power(T - 1, 1.5) + 1.5),
                    1,
                )
                + s.imm_regrets[a],
                0,
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T, 3) * p
            )
        s.reach = 0


class NoclipCFRSolver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = max(
                s.regrets[a] * np.power(T - 1, 1.5) / (np.power(T - 1, 1.5) + 1.5)
                + s.imm_regrets[a],
                0,
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T, 3) * p
            )
        s.reach = 0


class AutoCFR4Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        imm_regrets = np.array(list(s.imm_regrets.values()))
        regrets = np.array(list(s.regrets.values()))

        h = (
            np.mean((regrets >= 0).astype(np.float64), axis=0, keepdims=True)
            * np.power(T, 1.5)
            / (np.power(T, 1.5) + 1)
        )
        regrets = (
            regrets
            * (
                (regrets >= 0).astype(np.float64)
                * np.power(T, 1.5)
                / (np.power(T, 1.5) + 1)
                + (regrets < 0).astype(np.float64) * h
            )
            + imm_regrets
        )
        for i, a in enumerate(s.regrets.keys()):
            s.regrets[a] = regrets[i]
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = s.cum_policy[a] * np.power((T - 1) / T, 2) + T * p
        s.reach = 0


class AutoCFRSSolver(CFRSolver):
    def normalize(self, vec):
        size = vec.shape[0]
        p_sum = np.sum(vec)
        if p_sum == 0:
            vec = np.ones(size, dtype=np.float64) / size
        else:
            vec = vec / p_sum
        return vec

    def cumulate_regret(self, s):
        T = float(self.iter_count)
        imm_regrets = np.array(list(s.imm_regrets.values()))
        regrets = np.array(list(s.regrets.values()))
        reach_probs = s.reach

        a = imm_regrets + reach_probs
        b = np.maximum(-0.1, a)
        c = -0.01 * a
        d = b + 2
        e = c * a
        f = self.normalize(d)
        g = self.normalize(e)
        h = np.power(f, T)
        i = g * d
        j = self.normalize(h)
        k = e + i
        l = np.minimum(j, k)
        s.ins_regrets = {}
        for i, a in enumerate(s.regrets.keys()):
            s.regrets[a] = l[i]
            s.ins_regrets[a] = s.imm_regrets[a]
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        ins_regrets = np.array(list(s.ins_regrets.values()))
        cumu_strategy = np.array(list(s.cum_policy.values()))

        a = np.maximum(0, ins_regrets)
        b = self.normalize(a)
        c = b / 1
        d = c + b
        e = (d >= 1.5).astype(np.float64)
        f = np.maximum(e, cumu_strategy)
        g = b + f
        for i, a in enumerate(s.regrets.keys()):
            s.cum_policy[a] = g[i]
        s.reach = 0

    def update_current_policy(self, s):
        cumu_regret = np.array(list(s.regrets.values()))
        cumu_strategy = np.array(list(s.cum_policy.values()))
        a = np.maximum(cumu_strategy, cumu_regret)
        b = a / 0.001
        c = b + 0.1
        d = self.normalize(c)
        for i, a in enumerate(s.regrets.keys()):
            s.policy[a] = d[i]
