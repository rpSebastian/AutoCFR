# Copyright (c) 2019 Eric Steinberger
# Inspiration of architecture from DeepStack-Leduc (https://github.com/lifrordi/DeepStack-Leduc/tree/master/Source)

import numpy as np
from pathlib import Path
import pickle
from PokerRL.game.Poker import Poker
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game._.tree._.nodes import PlayerActionNode
from PokerRL.game._.tree._.card_to_string_conversion import card_to_string

class ValueFiller:

    def __init__(self, tree):
        self._tree = tree
        self._env_bldr = tree.env_bldr
        self._env = self._env_bldr.get_new_env(is_evaluating=True)

        # This only works for 1-Card games!
        self._eq_const = (self._env_bldr.rules.N_CARDS_IN_DECK / (self._env_bldr.rules.N_CARDS_IN_DECK - 1))

    def compute_cf_values_heads_up(self, node):
        """
        The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
        Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
        """
        assert self._tree.n_seats == 2

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)
            # Fold
            if node.action == Poker.FOLD:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_fold_eq_final_street(node=node)
                else:
                    equity = self._get_fold_eq_preflop(node=node)

            # Check / Call
            else:
                if node.env_state[EnvDictIdxs.current_round] == Poker.FLOP:
                    equity = self._get_call_eq_final_street(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d])

                else:  # preflop
                    equity = self._get_call_eq_preflop(node=node)

            # set boardcards to 0
            for c in self._env_bldr.lut_holder.get_1d_cards(node.env_state[EnvDictIdxs.board_2d]):
                if c != Poker.CARD_NOT_DEALT_TOKEN_1D:
                    equity[:, c] = 0.0

            node.ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2
            node.ev_br = np.copy(node.ev)
        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)
            ev_br_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                         dtype=np.float32)

            for i, child in enumerate(node.children):
                self.compute_cf_values_heads_up(node=child)
                ev_all_actions[i] = child.ev
                ev_br_all_actions[i] = child.ev_br

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.ev = np.sum(ev_all_actions, axis=0)
                node.ev_br = np.sum(ev_br_all_actions, axis=0)

            else:
                node.ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
                node.ev_br = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next

                node.ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.ev[opp] = np.sum(ev_all_actions[:, opp], axis=0)

                node.ev_br[opp] = np.sum(ev_br_all_actions[:, opp], axis=0)
                node.ev_br[plyr] = np.max(ev_br_all_actions[:, plyr], axis=0)

                node.br_a_idx_in_child_arr_for_each_hand = np.argmax(ev_br_all_actions[:, plyr], axis=0)

        # weight ev by reach prob
        node.ev_weighted = node.ev * node.reach_probs
        node.ev_br_weighted = node.ev_br * node.reach_probs
        assert np.allclose(np.sum(node.ev_weighted), 0, atol=0.001), np.sum(node.ev_weighted)  # Zero Sum check

        node.epsilon = node.ev_br_weighted - node.ev_weighted
        node.exploitability = np.sum(node.epsilon, axis=1)

    def _get_fold_eq_preflop(self, node):
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        for p in range(self._tree.n_seats):
            opp = 1 - p

            # sum reach probs for all hands and subtracts the reach prob of the hand player p holds batched for all
            equity[p] = np.sum(node.reach_probs[opp]) - node.reach_probs[opp]

        equity[node.p_id_acted_last] *= -1
        return equity * self._eq_const

    def _get_fold_eq_final_street(self, node):
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        for p in range(self._tree.n_seats):
            opp = 1 - p

            # sum reach probs for all hands and subtracts the reach prob of the hand player p holds batched for all
            equity[p] = np.sum(node.reach_probs[opp]) - node.reach_probs[opp]

        equity[node.p_id_acted_last] *= -1
        return equity * self._eq_const

    def _get_call_eq_final_street(self, reach_probs, board_2d):
        """
        Returns:
            equity: negative=lose. positive=win. 0=50%/50%

        """
        c = self._env_bldr.lut_holder.get_1d_cards(board_2d)[0]

        assert c != Poker.CARD_NOT_DEALT_TOKEN_1D

        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
        handranks = np.empty(shape=self._env_bldr.rules.RANGE_SIZE, dtype=np.int32)

        for h in range(self._env_bldr.rules.RANGE_SIZE):
            handranks[h] = self._env.get_hand_rank(
                board_2d=board_2d,
                hand_2d=self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h))

        for p in range(self._tree.n_seats):
            opp = 1 - p
            for h in range(self._env_bldr.rules.RANGE_SIZE):
                if h != c:
                    for h_opp in range(self._env_bldr.rules.RANGE_SIZE):
                        if h_opp != h and h_opp != c:
                            # when same handrank, would be += 0
                            if handranks[h] > handranks[h_opp]:
                                equity[p, h] += reach_probs[opp, h_opp]
                            elif handranks[h] < handranks[h_opp]:
                                equity[p, h] -= reach_probs[opp, h_opp]

        assert np.allclose(equity[:, c], 0)
        return equity * self._eq_const

    def _get_call_eq_preflop(self, node):
        # very Leduc specific
        equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

        for c in range(self._env_bldr.rules.N_CARDS_IN_DECK):
            """ ._get_call_eq() returns 0 for blocked hands, so we are summing 5 hands for each board. """
            _board_1d = np.array([c], dtype=np.int8)
            _board_2d = self._env_bldr.lut_holder.get_2d_cards(_board_1d)
            _reach_probs = np.copy(node.reach_probs)
            _reach_probs[:, c] = 0

            equity += self._get_call_eq_final_street(reach_probs=_reach_probs, board_2d=_board_2d)

        # mean :: using (N_CARDS_IN_DECK - 2) because two boards are blocked (by agent's and opp's cards)
        equity /= (self._env_bldr.rules.N_CARDS_IN_DECK - 2)
        return equity  # * self._eq_const

class ValueFillerHUNL(ValueFiller):
    def __init__(self, tree):
        self._tree = tree 
        self._env_bldr = tree.env_bldr
        self._env = self._env_bldr.get_new_env(is_evaluating=True)

        # This only works for 1-Card games!
        self._eq_const = (self._env_bldr.rules.N_CARDS_IN_DECK / (self._env_bldr.rules.N_CARDS_IN_DECK - 1))
        # self._block_matrix = self._create_block_matrix()
        self.root_path = Path(__file__).parent.parent.parent.parent.parent.parent
        self._block_matrix = np.load(self.root_path / "block_matrix.npy")
        self._texas_lookup = np.load(self.root_path / 'texas_lookup.npy')
        self.load_matrix_dict()

    def evaluate(self, hands, mask=None):
        rank = self._texas_lookup[ hands[ : , 0 ] + 54 ]
        for c in range(1, hands.shape[1]):
            rank = self._texas_lookup[ hands[ : , c ] + rank + 1 ]
        if mask is not None:
            rank *= mask
        # rank *= -1
        return rank

    def get_hand_rank(self, board_2d, hand_2d):
        # Libratus id to pystack id
        board_1d = [self._env.id_2d_to_str_dict[tuple(b)] for b in board_2d]
        board_id = [card_to_string.string_to_card(b) for b in board_1d]

        hand_1d = [self._env.id_2d_to_str_dict[tuple(b)] for b in hand_2d]
        hand_id = [card_to_string.string_to_card(b) for b in hand_1d]

        hands = np.array([board_id + hand_id])
        hand_rank = self.evaluate(hands)[0]
        return hand_rank

    def _create_block_matrix(self):
        ''' Creates boolean mask matrix for hands, that cannot be available
        	if particular cards where used. (ex: if hand1 is 'KsQs', then all
        	hand combinations with 'Ks' or 'Qs' should not be available)
        @return [I,I] :boolean mask for possible hands
        '''
        out = np.ones([1326, 1326], dtype=bool)
        for p1_card1 in range(52):
            for p1_card2 in range(p1_card1+1, 52):
                p1_card1_str = self._env.id_1d_to_str_dict[p1_card1]
                p1_card2_str = self._env.id_1d_to_str_dict[p1_card2]
                hand_str = p1_card1_str + p1_card2_str
                p1_idx = self._env.str_to_hand_id_dict[hand_str]
                # p1_idx = self._env.lut_holder.get_range_idx_from_hole_cards(self._env_bldr.lut_holder.get_2d_cards(np.array([p1_card1, p1_card2])))
                for p2_card1 in range(52):
                    for p2_card2 in range(p2_card1+1, 52):
                        p2_card1_str = self._env.id_1d_to_str_dict[p2_card1]
                        p2_card2_str = self._env.id_1d_to_str_dict[p2_card2]
                        hand_str = p2_card1_str + p2_card2_str
                        p2_idx = self._env.str_to_hand_id_dict[hand_str]
                        if p1_card1 == p2_card1 or p1_card1 == p2_card2 or \
                            p1_card2 == p2_card1 or p1_card2 == p2_card2:
                            out[p1_idx, p2_idx] = 0
                            out[p2_idx, p1_idx] = 0
        np.save(self.root_path / "block_matrix.npy", out)
        return out

    def set_board(self, node):
        board_2d = node.env_state[EnvDictIdxs.board_2d]
        board_1d = list(self._env_bldr.lut_holder.get_1d_cards(board_2d))
        hash_board_1d = tuple(sorted(board_1d))
        if hash_board_1d in self.fold_matrix_dict and hash_board_1d in self.equity_matrix_dict:
            self.fold_matrix = self.fold_matrix_dict[hash_board_1d]
            self.equity_matrix = self.equity_matrix_dict[hash_board_1d]
            return
        print(node)
        round = node.env_state[EnvDictIdxs.current_round]
        if round == Poker.RIVER:
            self.equity_matrix = np.zeros([1326, 1326], dtype=np.float32)
            self._set_last_round_equity_matrix(self.equity_matrix, board_2d)    
            self._handle_blocking_cards(self.equity_matrix, board_1d)
        elif round == Poker.TURN:
            self.equity_matrix = np.zeros([1326, 1326], dtype=np.float32)
            for card_1d in range(52):
                if card_1d not in board_1d:
                    card_str = self._env.id_1d_to_str_dict[card_1d]
                    card_2d = self._env.str_to_id_dict[card_str]
                    new_board_2d = board_2d.copy()
                    new_board_2d[-1] = card_2d
                    new_board_1d = list(self._env_bldr.lut_holder.get_1d_cards(new_board_2d))
                    temp_equity_matrix = np.zeros([1326, 1326], dtype=np.float32)
                    self._set_last_round_equity_matrix(temp_equity_matrix, new_board_2d)
                    self._handle_blocking_cards(temp_equity_matrix, new_board_1d)
                    self.equity_matrix += temp_equity_matrix
            self.equity_matrix /= 48
        else:
            raise NotImplementedError
        self.fold_matrix = np.ones([1326, 1326], dtype=np.float32)
        self._handle_blocking_cards(self.fold_matrix, board_1d)

        self.fold_matrix_dict[hash_board_1d] = self.fold_matrix
        self.equity_matrix_dict[hash_board_1d] = self.equity_matrix
        self.save_matrix_dict()
    
    def load_matrix_dict(self):
        file = self.root_path / "fold_matrix_dict.pkl"
        if not file.exists():
            self.fold_matrix_dict = {}
            self.equity_matrix_dict = {}
        else:
            with open(self.root_path / "fold_matrix_dict.pkl", "rb") as f:
                self.fold_matrix_dict = pickle.load(f)
            with open(self.root_path / "equity_matrix_dict.pkl", "rb") as f:
                self.equity_matrix_dict = pickle.load(f)

    def save_matrix_dict(self):
        with open(self.root_path / "fold_matrix_dict.pkl", "wb") as f:
            pickle.dump(self.fold_matrix_dict, f)
        with open(self.root_path / "equity_matrix_dict.pkl", "wb") as f:
            pickle.dump(self.equity_matrix_dict, f)

    def _handle_blocking_cards(self, matrix, board_1d):
        ''' Zeroes entries in an equity matrix that correspond to invalid hands.
        	A hand is invalid if it shares any cards with the board
            @param: [I,I] :matrix that needs to be modified
            @param: [0-5] :vector of board cards
        '''
        possible_hand_indexes = self._get_possible_hands_mask(board_1d)
        matrix[:,:] *= possible_hand_indexes.reshape([1, 1326])
        matrix[:,:] *= possible_hand_indexes.reshape([1326, 1])
        matrix[:,:] *= self._block_matrix

    def _get_possible_hands_mask(self, board_1d):
        out = np.zeros([1326], dtype=np.float32)
        used = np.zeros([52], dtype=bool)
        for card in board_1d:
            if card != -127:
                used[ card ] = 1
        for card1 in range(52):
            if not used[card1]:
                for card2 in range(card1+1, 52):
                    if not used[card2]:
                        card1_str = self._env.id_1d_to_str_dict[card1]
                        card2_str = self._env.id_1d_to_str_dict[card2]
                        hand_str = card1_str + card2_str
                        hand_index = self._env.str_to_hand_id_dict[hand_str]
                        out[ hand_index ] = 1
        return out

    def _set_last_round_equity_matrix(self, equity_matrix, board_2d):
        ''' Constructs the matrix that turns player ranges into showdown equity.
        	Gives the matrix `A` such that for player ranges `x` and `y`, `x'Ay`
        	is the equity for the first player when no player folds
        @param: [I,I] :matrix that needs to be modified
        @param: [I,I] :board_cards a non-empty vector of board cards
        '''
        # batch eval with only single batch, because its last round
        strength = self.evaluate_board(board_2d)
        # handling hand stregths (winning probs)
        strength_view_1 = strength.reshape([1326, 1])
        strength_view_2 = strength.reshape([1, 1326])
        equity_matrix[:,:] = (strength_view_1 > strength_view_2).astype(int)
        equity_matrix[:,:] -= (strength_view_1 < strength_view_2).astype(int)

    def test_handranks(self, handranks, board_2d):
        out = []
        for h, handrank in enumerate(handranks):
            hand_str = self._env.hand_id_to_str_dict[h]
            board_1d = [self._env.id_2d_to_str_dict[tuple(b)] for b in board_2d]
            board_1d_str = ' '.join(board_1d)
            out.append((hand_str, board_1d_str, handrank)) 
        out.sort(key=lambda x: (x[2], x[0]))
        for o in out:
            print(o[0], o[1], o[2])
            # if o[0] == "TsAs":

    def evaluate_board(self, board_2d):
        handranks = np.empty(shape=self._env_bldr.rules.RANGE_SIZE, dtype=np.int32)
        for h in range(self._env_bldr.rules.RANGE_SIZE):
            hand_1d = list(self._env_bldr.lut_holder.get_1d_cards(
                                    self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h)))
            board_1d = list(self._env_bldr.lut_holder.get_1d_cards(board_2d))
            if self._check_conflict(hand_1d, board_1d):
                handranks[h] = 0
            else:
                handranks[h] = self.get_hand_rank(
                    board_2d=board_2d,
                    hand_2d=self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h))
        # self.test_handranks(handranks, board_2d)
        return handranks

    def compute_cf_values_heads_up(self, node):
        """
        The functionality is extremely simplified compared to n-agent evaluations and made for HU Leduc only!
        Furthermore, this BR implementation is *VERY* inefficient and not suitable for anything much bigger than Leduc.
        """
        self.set_board(node)
        assert self._tree.n_seats == 2

        if node.is_terminal:
            assert node.strategy is None
        else:
            assert node.strategy.shape == (self._env_bldr.rules.RANGE_SIZE, len(node.children),)

        if node.is_terminal:
            """
            equity: -1*reach=always lose. 1*reach=always win. 0=50%/50%
            """
            assert isinstance(node, PlayerActionNode)
            # Fold
            if node.action == Poker.FOLD:
                equity = self._get_fold_eq(reach_probs=node.reach_probs, node=node)

            # Check / Call
            else:
                equity = self._get_call_eq(reach_probs=node.reach_probs,
                                                            board_2d=node.env_state[EnvDictIdxs.board_2d])

            node.ev = equity * node.env_state[EnvDictIdxs.main_pot] / 2
            node.ev_br = np.copy(node.ev)
        else:
            N_ACTIONS = len(node.children)
            ev_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                      dtype=np.float32)
            ev_br_all_actions = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE),
                                         dtype=np.float32)

            for i, child in enumerate(node.children):
                self.compute_cf_values_heads_up(node=child)
                ev_all_actions[i] = child.ev
                ev_br_all_actions[i] = child.ev_br

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                node.ev = np.sum(ev_all_actions, axis=0)
                node.ev_br = np.sum(ev_br_all_actions, axis=0)

            else:
                node.ev = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
                node.ev_br = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - node.p_id_acting_next

                node.ev[plyr] = np.sum(node.strategy.T * ev_all_actions[:, plyr], axis=0)
                node.ev[opp] = np.sum(ev_all_actions[:, opp], axis=0)

                node.ev_br[opp] = np.sum(ev_br_all_actions[:, opp], axis=0)
                node.ev_br[plyr] = np.max(ev_br_all_actions[:, plyr], axis=0)

                node.br_a_idx_in_child_arr_for_each_hand = np.argmax(ev_br_all_actions[:, plyr], axis=0)
        # weight ev by reach prob
        node.ev_weighted = node.ev * node.reach_probs
        node.ev_br_weighted = node.ev_br * node.reach_probs
        assert np.allclose(np.sum(node.ev_weighted), 0, atol=0.001), np.sum(node.ev_weighted)  # Zero Sum check

        node.epsilon = node.ev_br_weighted - node.ev_weighted
        node.exploitability = np.sum(node.epsilon, axis=1)

    def _get_fold_eq(self, reach_probs, node):
        equity = np.zeros([2, 1326], dtype=np.float32)
        equity[0] = np.matmul(self.fold_matrix, reach_probs[1, :])
        equity[1] = np.matmul(self.fold_matrix, reach_probs[0, :])
        equity[node.p_id_acted_last] *= -1
        return equity

    def _check_conflict(self, list_a, list_b):
        for item in list_a:
            if item in list_b:
                return True
        return False
    
    def _get_call_eq(self, reach_probs, board_2d):
        """
        Returns:
            equity: negative=lose. positive=win. 0=50%/50%

        """
        equity = np.zeros([2, 1326], dtype=np.float32)
        equity[0] = np.matmul(self.equity_matrix, reach_probs[1, :])
        equity[1] = np.matmul(self.equity_matrix, reach_probs[0, :])
        # board_1d_list = list(self._env_bldr.lut_holder.get_1d_cards(board_2d))

        # equity = np.zeros(shape=(self._tree.n_seats, self._env_bldr.rules.RANGE_SIZE), dtype=np.float32)
        # handranks = np.empty(shape=self._env_bldr.rules.RANGE_SIZE, dtype=np.int32)

        # for h in range(self._env_bldr.rules.RANGE_SIZE):
        #     handranks[h] = self._env.get_hand_rank(
        #         board_2d=board_2d,
        #         hand_2d=self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h))
        # # 判断手牌是否与公共牌冲突
        # for p in range(self._tree.n_seats):
        #     opp = 1 - p
        #     for h in range(self._env_bldr.rules.RANGE_SIZE):
        #         print(p, h)
        #         h_hand_1d_list = list(self._env_bldr.lut_holder.get_1d_cards(
        #                             self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h)))
                
        #         if self._check_conflict(h_hand_1d_list, board_1d_list):
        #             assert np.sum(self.equity_matrix[h, :]) == 0
        #             assert np.sum(self.equity_matrix[:, h]) == 0
        #             continue
        #         for h_opp in range(self._env_bldr.rules.RANGE_SIZE):
        #             h_opp_hand_1d_list = list(self._env_bldr.lut_holder.get_1d_cards(
        #                             self._env_bldr.lut_holder.get_2d_hole_cards_from_range_idx(range_idx=h_opp)))
        #             if self._check_conflict(h_opp_hand_1d_list, board_1d_list + h_hand_1d_list):
        #                 assert np.sum(self.equity_matrix[h, h_opp]) == 0
        #                 continue
                            
        #             # when same handrank, would be += 0
        #             if handranks[h] > handranks[h_opp]:
        #                 equity[p, h] += reach_probs[opp, h_opp]
        #             elif handranks[h] < handranks[h_opp]:
        #                 equity[p, h] -= reach_probs[opp, h_opp]
                    
        #             if handranks[h] < handranks[h_opp]:
        #                 assert self.equity_matrix[h, h_opp] == -1
        #             if handranks[h] > handranks[h_opp]:
        #                 assert self.equity_matrix[h, h_opp] == 1
        #             if handranks[h] == handranks[h_opp]:
        #                 assert self.equity_matrix[h, h_opp] == 0
        # print(equity)
        # print(equity_ans)
        # raise ValueError("")
        # assert np.allclose(equity[:, c], 0)
        return equity 
