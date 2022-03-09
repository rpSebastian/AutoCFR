# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.cfr._CFRBase import CFRBase as _CFRBase


class AutoCFRS(_CFRBase):

    def __init__(self,
                 name,
                 chief_handle,
                 game_cls,
                 agent_bet_set,
                 other_agent_bet_set=None,
                 starting_stack_sizes=None,
                 ):
        super().__init__(name=name,
                         chief_handle=chief_handle,
                         game_cls=game_cls,
                         starting_stack_sizes=starting_stack_sizes,
                         agent_bet_set=agent_bet_set,
                         other_agent_bet_set=other_agent_bet_set,
                         algo_name="AutoCFRS"
                         )
        self.reset()
    
    def normalize(self, vec):
        size = vec.shape[0]
        p_sum = np.sum(vec)
        if p_sum == 0:
            vec = np.ones(size, dtype=np.float32) / size
        else:
            vec = vec / p_sum
        return vec


    def normalize_matrix(self, x):
        y = np.zeros_like(x, dtype=np.float32)
        for a in range(x.shape[0]):
            y[a] = self.normalize(x[a])
        return y

    def _regret_formula_after_first_it(self, ev_all_actions, strat_ev, last_regrets, reach_probs):
        imm_regrets = ev_all_actions - strat_ev
        regrets = last_regrets
        T = self._iter_counter + 1
        reach_probs = np.expand_dims(reach_probs, axis=1)
        a = imm_regrets + reach_probs
        b = np.maximum(-0.1, a)
        c = -0.01 * a
        d = b + 2
        e = c * a
        f = self.normalize_matrix(d)
        g = self.normalize_matrix(e)
        h = np.power(f, T)
        i = g * d
        j = self.normalize_matrix(h)
        k = e + i
        l = np.minimum(j, k)
        regrets = l
        # regrets = np.maximum(np.minim um(regrets, T) + T * imm_regrets, 0)
        # regrets = np.maximum(np.minimum(regrets * np.power(T - 1, 1.5) / (np.power(T - 1, 1.5) + 1.5), 1) + imm_regrets, 0)
        # regrets = self.normalize_matrix(regrets)
        return regrets

    def _regret_formula_first_it(self, ev_all_actions, strat_ev):
        return ev_all_actions - strat_ev

    def _compute_new_strategy(self, p_id):
        for t_idx in range(len(self._trees)):
            def _fill(_node):
                if _node.p_id_acting_next == p_id:
                    N = len(_node.children)
                    _capped_reg = np.maximum(_node.data["regret"], 0)
                    cumu_strategy = _node.data["avg_strat_sum"]
                    cumu_regret = _node.data["regret"]
                    a = np.maximum(cumu_strategy, cumu_regret)
                    b = a / 0.001
                    c = b + 0.1
                    _node.strategy = self.normalize_matrix(c)
                for c in _node.children:
                    _fill(c)

            _fill(self._trees[t_idx].root)

    def _add_strategy_to_average(self, p_id):
        def _fill(_node):
            if _node.p_id_acting_next == p_id:
                T = self._iter_counter + 1
                if self._iter_counter == 0:
                    _node.data["avg_strat_sum"] = np.zeros_like(_node.data["regret"])
                ins_regret = _node.data["ins_regret"]
                cumu_strategy = _node.data["avg_strat_sum"]
                a = np.maximum(0, ins_regret)                 
                b = self.normalize_matrix(a)
                c = b / 1
                d = c + b
                e = (d >= 1.5).astype(np.float32)
                f = np.maximum(e, cumu_strategy)
                g = b + f
                _node.data["avg_strat_sum"] = g
                
                _s = np.expand_dims(np.sum(_node.data["avg_strat_sum"], axis=1), axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    _node.data["avg_strat"] = np.where(_s == 0,
                                                       np.full(shape=len(_node.allowed_actions),
                                                               fill_value=1.0 / len(_node.allowed_actions)),
                                                       _node.data["avg_strat_sum"] / _s
                                                       )
                assert np.allclose(np.sum(_node.data["avg_strat"], axis=1), 1, atol=0.0001)

            for c in _node.children:
                _fill(c)

        for t_idx in range(len(self._trees)):
            _fill(self._trees[t_idx].root)


    def _compute_regrets(self, p_id):

        for t_idx in range(len(self._trees)):
            def __compute_evs(_node):
                # EV of each action
                N_ACTIONS = len(_node.children)
                ev_all_actions = np.zeros(shape=(self._env_bldrs[t_idx].rules.RANGE_SIZE, N_ACTIONS), dtype=np.float32)
                for i, child in enumerate(_node.children):
                    ev_all_actions[:, i] = child.ev[p_id]

                # EV if playing by curr strat
                strat_ev = _node.ev[p_id]
                strat_ev = np.expand_dims(strat_ev, axis=-1).repeat(N_ACTIONS, axis=-1)

                return strat_ev, ev_all_actions

            def _fill_after_first(_node):
                if _node.p_id_acting_next == p_id:
                    strat_ev, ev_all_actions = __compute_evs(_node=_node)
                    _node.data["regret"] = self._regret_formula_after_first_it(ev_all_actions=ev_all_actions,
                                                                               strat_ev=strat_ev,
                                                                               last_regrets=_node.data["regret"],
                                                                               reach_probs=_node.reach_probs[p_id])
                    _node.data["ins_regret"] = ev_all_actions - strat_ev
                for c in _node.children:
                    _fill_after_first(c)

            def _fill_first(_node):
                if _node.p_id_acting_next == p_id:
                    strat_ev, ev_all_actions = __compute_evs(_node=_node)

                    _node.data["regret"] = self._regret_formula_first_it(ev_all_actions=ev_all_actions,
                                                                         strat_ev=strat_ev)
                    _node.data["ins_regret"] = ev_all_actions - strat_ev
                for c in _node.children:
                    _fill_first(c)

            if self._iter_counter == 0:
                _fill_first(self._trees[t_idx].root)
            else:
                _fill_after_first(self._trees[t_idx].root)
