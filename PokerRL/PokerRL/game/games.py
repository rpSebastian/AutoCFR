# Copyright (c) 2019 Eric Steinberger


"""
A collection of Poker games often used in computational poker research.
"""
import numpy as np
from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules import HoldemRules, LeducRules, FlopHoldemRules, BigLeducRules
from PokerRL.game._.rl_env.poker_types.DiscretizedPokerEnv import DiscretizedPokerEnv
from PokerRL.game._.rl_env.poker_types.LimitPokerEnv import LimitPokerEnv
from PokerRL.game._.rl_env.poker_types.NoLimitPokerEnv import NoLimitPokerEnv
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs, PlayerDictIdxs
from PokerRL.game._.rl_env.base._Deck import DeckOfCards

# """""""""""""""
# Leduc Family
# """""""""""""""
class StandardLeduc(LeducRules, LimitPokerEnv):
    """
    Leduc Hold'em is a very small poker game meant for fast experimentation with new algorithms. It is played with 3
    ranks and 2 suits. Typically players place an ante of 1, the small_bet is 2, and the big_bet is 4.
    """

    RULES = LeducRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 2,
        Poker.FLOP: 2,
    }

    SMALL_BLIND = 0
    BIG_BLIND = 0
    ANTE = 1
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 13

    EV_NORMALIZER = 1000.0 / ANTE  # Milli Antes
    WIN_METRIC = Poker.MeasureAnte

    ROUND_WHERE_BIG_BET_STARTS = Poker.FLOP

    def __init__(self, env_args, lut_holder, is_evaluating):
        LeducRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)


class BigLeduc(BigLeducRules, LimitPokerEnv):
    RULES = BigLeducRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 6,
        Poker.FLOP: 6,
    }

    SMALL_BLIND = 0
    BIG_BLIND = 0
    ANTE = 1
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 100

    EV_NORMALIZER = 1000.0 / ANTE  # Milli Antes
    WIN_METRIC = Poker.MeasureAnte

    ROUND_WHERE_BIG_BET_STARTS = Poker.FLOP

    def __init__(self, env_args, lut_holder, is_evaluating):
        BigLeducRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)

class MidLeduc(BigLeducRules, LimitPokerEnv):
    RULES = BigLeducRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 2,
        Poker.FLOP: 2,
    }

    SMALL_BLIND = 0
    BIG_BLIND = 0
    ANTE = 1
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 100

    EV_NORMALIZER = 1000.0 / ANTE  # Milli Antes
    WIN_METRIC = Poker.MeasureAnte

    ROUND_WHERE_BIG_BET_STARTS = Poker.FLOP

    def __init__(self, env_args, lut_holder, is_evaluating):
        BigLeducRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)


class NoLimitLeduc(LeducRules, NoLimitPokerEnv):
    """
    A variant of Leduc with no bet-cap in the no-limit format. It uses blinds instead of antes.
    """

    RULES = LeducRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        LeducRules.__init__(self)
        NoLimitPokerEnv.__init__(self,
                                 env_args=env_args,
                                 lut_holder=lut_holder,
                                 is_evaluating=is_evaluating)


class DiscretizedNLLeduc(LeducRules, DiscretizedPokerEnv):
    """
    Discretized version of No-Limit Leduc Hold'em (i.e. agents can only select from a predefined set of betsizes)
    """

    RULES = LeducRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        LeducRules.__init__(self)
        DiscretizedPokerEnv.__init__(self,
                                     env_args=env_args,
                                     lut_holder=lut_holder,
                                     is_evaluating=is_evaluating)


# """""""""""""""
# Hold'em Family
# """""""""""""""
class LimitHoldem(HoldemRules, LimitPokerEnv):
    """
    Fixed-Limit Texas Hold'em is a long-standing benchmark game that has been essentially solved by Bowling et al
    (http://science.sciencemag.org/content/347/6218/145) using an efficient distributed implementation of CFR+, an
    optimized version of regular CFR.
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 4,
        Poker.FLOP: 4,
        Poker.TURN: 4,
        Poker.RIVER: 4,
    }
    ROUND_WHERE_BIG_BET_STARTS = Poker.TURN

    SMALL_BLIND = 1
    BIG_BLIND = 2
    ANTE = 0
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 48

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)


class NoLimitHoldem(HoldemRules, NoLimitPokerEnv):
    """
    No-Limit Texas Hold'em is the largest poker game in which AI beat humans as of 31.08.2018. It has been the focus in
    work such as DeepStack (https://arxiv.org/abs/1701.01724) and Libratus
    (http://science.sciencemag.org/content/early/2017/12/15/science.aao1733).
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        NoLimitPokerEnv.__init__(self,
                                 env_args=env_args,
                                 lut_holder=lut_holder,
                                 is_evaluating=is_evaluating)


class DiscretizedNLHoldem(HoldemRules, DiscretizedPokerEnv):
    """
    Discretized version of No-Limit Texas Hold'em (i.e. agents can only select from a predefined set of betsizes)
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        DiscretizedPokerEnv.__init__(self,
                                     env_args=env_args,
                                     lut_holder=lut_holder,
                                     is_evaluating=is_evaluating)


class Flop5Holdem(FlopHoldemRules, LimitPokerEnv):
    RULES = FlopHoldemRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 2,  # is actually 1, but BB counts as a raise in this codebase
        Poker.FLOP: 2,
    }
    ROUND_WHERE_BIG_BET_STARTS = Poker.TURN

    UNITS_SMALL_BET = None
    UNITS_BIG_BET = None

    FIRST_ACTION_NO_CALL = True

    def __init__(self, env_args, lut_holder, is_evaluating):
        FlopHoldemRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)

    def _adjust_raise(self, raise_total_amount_in_chips):
        return self.get_fraction_of_pot_raise(fraction=1.0, player_that_bets=self.current_player)


from pathlib import Path
def read_subgame_file(subgame_id):
    file = Path(__file__).parent.parent.parent / "LibratusEndgames" / "subgame{}.txt".format(subgame_id)
    with file.open("r") as f:
        for line in f.readlines():
            if "round" in line:
                round = int(line.strip().split(" ")[1])
            if "board" in line:
                board = line.strip().split(" ")[1]
                board = [board[i * 2: (i + 1) * 2] for i in range(len(board) // 2)]
                    
            if "pot" in line:
                pot = int(line.strip().split(" ")[1])
            if "reach" in line:
                reach = list(map(float, line.strip().split(" ")[1:]))
    return round, board, pot, reach[:1326], reach[1326:]

class DiscretizedNLHoldemSubGame(DiscretizedNLHoldem):
    CURRENT_ROUND = NotImplemented
    SUBGAME_ID = NotImplemented

    def __init__(self, env_args, lut_holder, is_evaluating):
        super().__init__(env_args, lut_holder, is_evaluating)
        self.build_str_to_id_dict()
        self.build_str_to_hand_id_dict()
        self.build_libratus_hand_id_to_str_dict()
        self.create_root_env_state()
    
    def build_str_to_id_dict(self):
        self.str_to_id_dict = {}
        self.id_2d_to_str_dict = {}
        self.id_1d_to_str_dict = {}
        for i in range(4):
            for j in range(13):
                cardid = [j, i]
                cardstr = self.cards2str([cardid])[:2]
                self.str_to_id_dict[cardstr] = cardid
                self.id_2d_to_str_dict[(j, i)] = cardstr
                self.id_2d_to_str_dict[(i, j)] = cardstr

                card_1d = self.lut_holder.get_1d_cards(np.array([[j, i]]))[0]
                self.id_1d_to_str_dict[card_1d] = cardstr

    def build_str_to_hand_id_dict(self):
        self.str_to_hand_id_dict = {}
        self.hand_id_to_str_dict = {}
        card_strs = list(self.str_to_id_dict.keys())
        for i in range(52):
            for j in range(i + 1, 52):
                card1_str = card_strs[i]
                card2_str = card_strs[j]

                card1_id = self.str_to_id_dict[card1_str]
                card2_id = self.str_to_id_dict[card2_str]
                if card1_id > card2_id:
                    card1_id, card2_id = card2_id, card1_id
                hand_2d_id = np.array([card1_id, card2_id])
                hand_id = self.lut_holder.get_range_idx_from_hole_cards(hand_2d_id)
                hand_str = card1_str + card2_str
                self.str_to_hand_id_dict[hand_str] = hand_id
                self.hand_id_to_str_dict[hand_id] = hand_str
                hand_str = card2_str + card1_str
                self.str_to_hand_id_dict[hand_str] = hand_id
    
    def build_libratus_hand_id_to_str_dict(self):
        self.libratus_hand_id_to_str_dict = {}
        card_ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        card_suits = ["s", "h", "d", 'c']
        card_strs = []
        for card_rank in card_ranks:
            for card_suit in card_suits:
                card_str = card_rank + card_suit
                card_strs.append(card_str)
        count = 0
        for i in range(52):
            for j in range(i + 1, 52):
                card1_str = card_strs[i]
                card2_str = card_strs[j]
                hand_str = card1_str + card2_str
                libratus_hand_id = count
                count += 1
                self.libratus_hand_id_to_str_dict[libratus_hand_id] = hand_str

    def libratus_reach_to_pokerRL_reach(self, libratus_reach):
        pokerRL_reach = [0 for i in range(1326)]
        for libratus_hand_id in range(1326):
            hand_str = self.libratus_hand_id_to_str_dict[libratus_hand_id]
            pokerRL_hand_id = self.str_to_hand_id_dict[hand_str]
            pokerRL_reach[pokerRL_hand_id] = libratus_reach[libratus_hand_id]
        return pokerRL_reach

    def create_root_env_state(self):
        round, board, pot, reach1, reach2 = read_subgame_file(self.SUBGAME_ID)
        deck = DeckOfCards(num_suits=4, num_ranks=13)
        board = np.array([self.str_to_id_dict[card_str] for card_str in board] + [[-127, -127] for _ in range(5 - len(board))])
        hand1 = np.array([[0, 0], [0, 1]])
        hand2 = np.array([[0, 2], [0, 3]])
        for cards in [board, hand1, hand2]:
            deck.remove_cards(cards)
        main_pot = pot
        stack = 20000
        self.root_env_state = {
            EnvDictIdxs.is_evaluating: True,
            EnvDictIdxs.current_round: self.CURRENT_ROUND,
            EnvDictIdxs.side_pots: [0, 0],
            EnvDictIdxs.main_pot: main_pot,  # int by value
            EnvDictIdxs.board_2d: board,  # np array
            EnvDictIdxs.last_action: [1, 2, 1],
            EnvDictIdxs.capped_raise: None,
            EnvDictIdxs.current_player: 1,  # idx in _env.seats
            EnvDictIdxs.last_raiser: None,
            EnvDictIdxs.deck: deck.state_dict(),
            EnvDictIdxs.n_actions_this_episode: 0,  # int
            EnvDictIdxs.seats:
            [
                {
                    PlayerDictIdxs.seat_id: 0,
                    PlayerDictIdxs.hand: hand1,  # np array
                    PlayerDictIdxs.hand_rank: None,  # int by value
                    PlayerDictIdxs.stack: stack - main_pot // 2,  # int by value
                    PlayerDictIdxs.current_bet: 0,  # int by value
                    PlayerDictIdxs.is_allin: False,  # bool by value
                    PlayerDictIdxs.folded_this_episode: False,  # bool by value
                    PlayerDictIdxs.has_acted_this_round: False,  # bool by value
                    PlayerDictIdxs.side_pot_rank: -1  # int by value
                },
                {
                    PlayerDictIdxs.seat_id: 1,
                    PlayerDictIdxs.hand: hand2,  # np array
                    PlayerDictIdxs.hand_rank: None,  # int by value
                    PlayerDictIdxs.stack: stack - main_pot // 2,  # int by value
                    PlayerDictIdxs.current_bet: 0,  # int by value
                    PlayerDictIdxs.is_allin: False,  # bool by value
                    PlayerDictIdxs.folded_this_episode: False,  # bool by value
                    PlayerDictIdxs.has_acted_this_round: False,  # bool by value
                    PlayerDictIdxs.side_pot_rank: -1  # int by value
                }
            ],
            EnvDictIdxs.n_raises_this_round: 0
        }
        reach1 = self.libratus_reach_to_pokerRL_reach(reach1)
        reach2 = self.libratus_reach_to_pokerRL_reach(reach2)
        self.reach_probs = np.array([reach1, reach2], dtype=np.float32)
        self.reach_probs /= np.sum(self.reach_probs, axis=1, keepdims=True)

class DiscretizedNLHoldemSubGame4(DiscretizedNLHoldemSubGame):
    CURRENT_ROUND = 3
    SUBGAME_ID = 4
    
class DiscretizedNLHoldemSubGame3(DiscretizedNLHoldemSubGame):
    CURRENT_ROUND = 3
    SUBGAME_ID = 3

class DiscretizedNLHoldemSubGame2(DiscretizedNLHoldemSubGame):
    CURRENT_ROUND = 2
    SUBGAME_ID = 2
class DiscretizedNLHoldemSubGame1(DiscretizedNLHoldemSubGame):
    CURRENT_ROUND = 2
    SUBGAME_ID = 1


"""
register all new envs here!
"""
ALL_ENVS = [
    StandardLeduc,
    BigLeduc,
    MidLeduc,
    NoLimitLeduc,
    DiscretizedNLLeduc,
    LimitHoldem,
    NoLimitHoldem,
    DiscretizedNLHoldem,
    Flop5Holdem,
    DiscretizedNLHoldemSubGame4,
    DiscretizedNLHoldemSubGame3,
    DiscretizedNLHoldemSubGame2,
    DiscretizedNLHoldemSubGame1,
]
