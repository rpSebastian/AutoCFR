import copy
import numpy as np
from autocfr.vanilla_cfr.cfr import CFRSolver


class CFR_65_139436Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            h = (
                int(s.regrets[a] >= 0)
                * np.power(T - 1, 1.5)
                / (np.power(T - 1, 1.5) + 1)
            )
            m = (
                s.regrets[a] * (h + int(h < 0.01) / (0.1 + np.power(T - 1, 1.5)))
                + s.imm_regrets[a]
            )
            s.regrets[a] = min(m, np.exp(0.01))
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = s.cum_policy[a] * np.power((T - 1) / T, 2) + s.reach * p
        s.reach = 0


class CFR_34_108604Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = (
                min(
                    int(s.regrets[a] >= 0)
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


class CFR_34_108604aaSolver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            # if s.regrets[a] >= 0:
            s.regrets[a] = max(
                min(
                    s.regrets[a] * np.power(T - 1, 1.5) / (np.power(T - 1, 1.5) + 1.5),
                    1.1,
                )
                + s.imm_regrets[a],
                0,
            )
            # else:
            #     s.regrets[a] = max(s.regrets[a], 0) + s.imm_regrets[a]
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T, 3) * p
            )
        s.reach = 0


class CFR_34_108606Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            # if s.regrets[a] >= 0:
            s.regrets[a] = max(
                min(
                    s.regrets[a] * np.power(T - 1, 1.5) / (np.power(T - 1, 1.5) + 1.5),
                    1,
                )
                + s.imm_regrets[a],
                0,
            )
            # else:
            #     s.regrets[a] = max(s.regrets[a], 0) + s.imm_regrets[a]
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T, 3) * p
            )
        s.reach = 0


class CFR_34_108605Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            # if s.regrets[a] >= 0:
            s.regrets[a] = max(
                min(
                    s.regrets[a] * np.power(T - 1, 1.5) / (np.power(T - 1, 1.5) + 1.5),
                    1,
                )
                + s.imm_regrets[a],
                0,
            )
            # else:
            #     s.regrets[a] = max(s.regrets[a], 0) + s.imm_regrets[a]
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * (T - 1) / T + s.reach * np.power(T - 1, 3) * p
            )
        s.reach = 0


class CFR_68_121339Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            g = (
                int(s.regrets[a] >= 0)
                * np.power(T - 1, 1.5)
                / (np.power(T - 1, 1.5) + 1.5)
            )
            s.regrets[a] = min(
                s.regrets[a] * (g + int(g >= np.exp(0.1))) + s.imm_regrets[a], 1
            )
            # else:
            #     s.regrets[a] = max(s.regrets[a], 0) + s.imm_regrets[a]
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = s.cum_policy[a] * np.power((T - 1) / T, 2) + s.policy[
                a
            ] * (T - 1)
        s.reach = 0


class CFR_65_139436Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            h = (
                int(s.regrets[a] >= 0)
                * np.power(T - 1, 1.5)
                / (np.power(T - 1, 1.5) + 1)
            )
            s.regrets[a] = min(
                np.exp(0.01),
                s.regrets[a] * (h + int(h < 0.01) / (0.1 + np.power(T - 1, 1.5)))
                + s.imm_regrets[a],
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * np.power((T - 1) / T, 2) + s.policy[a] * s.reach
            )
        s.reach = 0


class CFR_65_475715Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = min(
                np.power(0.1, -0.1),
                s.regrets[a]
                * (
                    int(s.regrets[a] >= 0)
                    * np.power(T - 1, 1.5)
                    / (1.5 + 0.1 + np.power(T - 1, 1.5))
                    + int(s.regrets[a] < 0) / (0.1 + np.power(T - 1, 1.5))
                )
                + s.imm_regrets[a],
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = (
                s.cum_policy[a] * np.power((T - 1) / T, 2) + s.policy[a] * s.reach
            ) * ((T - 1) / T)
        s.reach = 0

class CFR_69_420733Solver(CFRSolver):
    def cumulate_regret(self, s):
        T = float(self.iter_count)
        for a in s.regrets.keys():
            s.regrets[a] = min(
                s.regrets[a]
                * (
                    int(s.regrets[a] >= 0)
                    * np.power(T - 1, 1.5)
                    / (np.power(T - 1, 1.5) + 1)
                    + min(0.01, int(s.regrets[a] >= 0))
                    * int(max(s.reach, -0.001) < 0.001)
                )
                + s.imm_regrets[a],
                1.5,
            )
            s.imm_regrets[a] = 0

    def cumulate_policy(self, s):
        T = float(self.iter_count)
        for a, p in s.policy.items():
            s.cum_policy[a] = np.power(
                np.exp(1.5), 0.002
            ) * s.cum_policy[a] * np.power((T - 1.5) / T, 2) - (
                s.reach * s.policy[a] / 0.1
            )
        s.reach = 0

    def normalize(self, s):
        policy_sum = 0
        for policy in s.policy.values():
            policy_sum += policy
        for a, policy in s.policy.items():
            if policy_sum == 0:
                s.policy[a] = 1 / s.na
            else:
                s.policy[a] = policy / policy_sum

    def update_current_policy(self, s):
        for a, regret in s.regrets.items():
            s.policy[a] = max(0, regret)
        c = copy.deepcopy(s)
        for a, policy in c.policy.items():
            c.policy[a] = min(0.1, c.policy[a])

        self.normalize(s)
        for a, policy in s.policy.items():
            s.policy[a] = max(-0.001, s.policy[a])
            s.policy[a] = max(s.policy[a], c.policy[a])
