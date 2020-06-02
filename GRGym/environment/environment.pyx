import logging
import typing
from copy import deepcopy
from typing import Dict, Tuple

import cython
import numpy as np

from GRGym.agent import BaseAgent
from .action_result cimport ActionResult
from .player cimport Player
from .player import Player
from .run cimport Run
from .set cimport Set
from .deadwood_counter cimport DeadwoodCounter
from libc.stdint cimport int64_t

@cython.final
cdef class Environment:
    def __init__(self, opponent_agent: BaseAgent):
        self.__player_1 = Player()
        self.__player_2 = Player()
        self.opponent_agent = opponent_agent
        self.__deck = np.arange(52, dtype=np.int8)
        self.__discard_pile = np.empty(52, dtype=np.int8)
        self.num_of_discard_cards = 0

        self.draw_phase = True  # Either draw phase or discard phase

    def reset_hand(self) -> np.ndarray:
        """
        Resets the environment in preparation for the next hand. Player scores are not reset.

        :return:
        """
        self.__player_1.reset_hand()
        self.__player_2.reset_hand()
        self.deck = np.arange(52, dtype=np.int8)
        np.random.shuffle(self.deck)
        self.discard_pile = np.empty(52, dtype=np.int8)
        self.num_of_discard_cards = 0

        self.draw_from_deck(self.player_1, 10)
        self.draw_from_deck(self.player_2, 10)
        self.add_first_discard_card()
        self.draw_phase = True
        return self.build_observations(self.player_1)

    def step(self, np.ndarray action) -> Tuple[np.ndarray, int]:
        # TODO FIRST CARD PRIVILEGE
        """
        Step through one interaction with the environment.

        :param action:
        A 56 length vector. First 52 elements correspond to the 52 cards.
        Next 2 elements correspond to draw from deck and draw from discard
        Last 2 elements correspond to knock and not knock

        :return:
        A tuple of results.
        The first element is the new observation vector.
        The second element is the reward.
        If the player won the hand, this is set to 1. If they won the match, this is set to 2.
        If they lost the hand, this is set to -1. If they lost the match, this is set to -2.
        Otherwise, this is set to 0.
        """
        cdef:
            ActionResult action_result
        if self.draw_phase:
            action_result = self.run_draw(action, self.player_1)
        else:  # Discard phase
            action_result = self.run_discard(action, self.player_1, is_player_1=True)
        self.draw_phase = not self.draw_phase
        if action_result != ActionResult.NO_CHANGE:
            self.reset_hand()
        return self.build_observations(self.player_1), action_result

    def run_draw(self, double[:] action, Player player):
        if len(self.deck) == 2:
            return ActionResult.DRAW
        if Environment.wants_to_draw_from_deck(action) or self.discard_pile_is_empty():
            self.draw_from_deck(player)
        else:
            self.draw_from_discard(player)
        return 0

    def run_discard(self, np.ndarray action, Player player, is_player_1: bool) -> int:
        if Environment.wants_to_knock(action) and Environment.is_gin(player):
            return self.score_big_gin(player)
        card_to_discard = self.get_card_to_discard(action, player)
        self.discard_card(player, card_to_discard)
        if Environment.wants_to_knock(action) and Environment.can_knock(player):
            if Environment.is_gin(player):
                return self.score_gin(player)
            else:
                score_delta = self.try_to_knock(player)
                if score_delta > 0:
                    return self.update_score(player, score_delta)
                else:
                    # Apply negative to return  relative to current player
                    return -self.update_score(self.opponents(player), -score_delta)
        if not is_player_1:
            return 0  # Avoid player 2 recursing back into player 1, want to only recurse once
        # Run player 2 draw and discard actions
        opponent_draw_action = self.opponent_agent.act(self.build_observations(self.player_2))
        self.run_draw(opponent_draw_action, self.player_2)
        opponent_discard_action = self.opponent_agent.act(self.build_observations(self.player_2))
        return self.run_discard(opponent_discard_action, self.player_2, False)

    @staticmethod
    def get_card_to_discard(np.ndarray action, Player player):
        # Sort the cards by how much the player prefers them. Negative makes it descending.
        # Keep it this way for benchmarking, convert to first method later.
        # actions_sorted = np.argsort(action[0:52])[::-1]
        actions_sorted = np.argsort(-action[0:52])
        # Sort the mask booleans (0 or 1 depending on if the player has them) by how much the player prefers them.
        card_mask_sorted = player.hand_mask()[actions_sorted]
        # Only take cards that the player has in their hand.
        cards_in_hand_sorted = actions_sorted[card_mask_sorted]
        # Take the most preferred card in the player's hand.
        card_to_discard = cards_in_hand_sorted[0]
        return card_to_discard

    def draw_from_deck(self, Player player, num_of_cards: int = 1):
        assert self.opponents(player)
        assert len(self.deck) >= num_of_cards
        for card_val in self.deck[:num_of_cards]:
            player.add_card_from_deck(card_val)
        self.deck = self.deck[num_of_cards:]
        return

    def draw_from_discard(self, Player player):
        cdef int8_t drawn_card = self.pop_from_discard_pile()
        cdef int8_t new_top_discard = player.NO_CARD if self.discard_pile_is_empty() else self.discard_pile[-1]
        player.add_card_from_discard(drawn_card, new_top_discard)
        self.opponents(player).report_opponent_drew_from_discard(drawn_card, new_top_discard)

    def build_observations(self, Player player) -> np.ndarray:
        """
        Builds observation array.
        52 elements for card_state
        1 element for current player score
        1 element for opponent score
        1 element for number of cards still left in deck
        1 element for whether or not the player is drawing
        1 element for whether or not the player is discarding
        Total of 57
        :param player:
        :return:
        """
        observation: np.ndarray = np.empty(57, np.int8)
        observation[0:52] = player.card_states
        observation[52] = player.score
        observation[53] = self.opponents(player).score
        observation[54] = len(self.deck)
        observation[55] = self.draw_phase
        observation[56] = not self.draw_phase
        return observation

    def discard_card(self, Player player, card_to_discard: int):
        previous_top = player.NO_CARD if self.discard_pile_is_empty() else self.discard_pile[-1]
        self.add_to_discard_pile(card_to_discard)
        player.discard_card(card_to_discard, previous_top)
        self.opponents(player).report_opponent_discarded(card_to_discard, previous_top)

    def try_to_knock(self, Player player):
        deadwood_counter = DeadwoodCounter(player.card_list())
        knocking_player_deadwood = deadwood_counter.deadwood()
        knocking_player_melds = deadwood_counter.melds()
        deadwood_counter = DeadwoodCounter(self.opponents(player).card_list())
        opponent_deadwood = deadwood_counter.deadwood()
        opponent_remaining_cards = deadwood_counter.remaining_cards()

        # layoff into runs first. This is because any cards laid off into runs will never block layoffs into sets,
        # while the opposite could happen.
        # Example: The knocking player has a run from 3-5 of diamonds and a set of 2 of Clubs, 2 of Hearts,
        # and 2 of Spades. The opponent has deadwood remaining cards of Ace of Diamonds and 2 of Diamonds.
        # If set layoffs run first, then 2 of Diamonds would be laid off, and the Ace of Diamonds would remain.
        # However, if run layoffs run first, then both the 2 of Diamonds and the Ace of Diamonds would be laid off.

        # try to layoff into runs
        # cdef Py_ssize_t i
        # cdef Run[:] run_melds=np.array([meld for meld in knocking_player_melds if isinstance(meld,Run)],dtype=object)
        cdef Run run
        cdef Set m_set

        cdef set run_melds = {meld for meld in knocking_player_melds if isinstance(meld, Run)}

        while True:
            did_something = False
            for run in run_melds:
                for card in run.connectable_cards():
                    if card in opponent_remaining_cards:
                        if card == run.start - 1:  # Append to the left
                            run.start -= 1
                        elif card == run.end + 1:
                            run.end += 1
                        else:
                            logging.error(
                                f"Card {card} was inserted into connectable_remaining_cards, but it is not an "
                                f"extension of any run. ")
                        opponent_remaining_cards.remove(card)
                        opponent_deadwood -= DeadwoodCounter.deadwood_val(card)
                        did_something = True
            if not did_something:
                break
        # try to layoff into sets
        set_melds: typing.Set[Set] = {deepcopy(meld) for meld in knocking_player_melds if isinstance(meld, Set)}
        while True:
            connectable_card_to_set: Dict[int, Set] = {card: m_set for m_set in set_melds for card in
                                                       m_set.connectable_cards()}
            connectable_remaining_cards = opponent_remaining_cards & connectable_card_to_set.keys()
            if not connectable_remaining_cards:
                break
            for card in connectable_remaining_cards:
                opponent_remaining_cards.remove(card)
                opponent_deadwood -= DeadwoodCounter.deadwood_val(card)

        # Check score difference
        if knocking_player_deadwood < opponent_deadwood:  # won hand
            return opponent_deadwood - knocking_player_deadwood
        else:  # undercut
            return -25 - (knocking_player_deadwood - opponent_deadwood)

    def score_big_gin(self, Player player) -> ActionResult:
        score_delta = DeadwoodCounter(self.opponents(player).card_list()).deadwood() + get_big_gin_bonus()
        return self.update_score(player, score_delta)

    def score_gin(self, Player player):
        score_delta = DeadwoodCounter(self.opponents(player).card_list()).deadwood() + get_gin_bonus()
        return self.update_score(player, score_delta)

    cdef ActionResult update_score(self, Player player, int64_t score_delta):
        player.score += score_delta
        if player.score >= get_score_limit():
            return ActionResult.WON_MATCH
        else:
            return ActionResult.WON_HAND

    @staticmethod
    cdef bint wants_to_draw_from_deck(double[:] action):
        return action[52] >= action[53]

    @staticmethod
    cdef bint wants_to_knock(double[:] action):
        return action[54] >= action[55]

    @staticmethod
    cdef bint is_gin(Player player):
        return DeadwoodCounter(player.card_list()).deadwood() == 0  # TODO refactor this expression separate method

    @staticmethod
    cdef bint can_knock(Player player):
        return DeadwoodCounter(player.card_list()).deadwood() <= 10

    cdef Player opponents(self, Player player):
        """
        Opponents is implemented as a function instead of a dictionary to allow for swapping of players.
        :param player:
        :return: The opponent player.
        """
        if player is self.__player_1:
            return self.player_2
        elif player is self.__player_2:
            return self.player_1
        else:
            raise ValueError('This player is not currently in the environment.')

    def __repr__(self):
        return f'{self.player_1.__repr__()}\n' \
               f'{self.player_2.__repr__()}\n' \
               f'Opponent Agent: {self.opponent_agent.__class__}\n' \
               f'Deck: {self.deck}\n' \
               f'Discard: {self.discard_pile}\n' \
               f'{"Draw" if self.draw_phase else "Discard"} Phase\n'

    cdef bint discard_pile_is_empty(self):
        return len(self.discard_pile) == 0

    cdef void add_first_discard_card(self):
        self.add_to_discard_pile(self.deck[0])
        self.__deck = self.__deck[1:]

    @property
    def deck(self):
        return np.asarray(self.__deck, dtype=np.int8)

    @deck.setter
    def deck(self, new_deck):
        self.__deck = new_deck.copy()

    @property
    def discard_pile(self):
        return np.asarray(self.__discard_pile[0:self.num_of_discard_cards], dtype=np.int8)

    @discard_pile.setter
    def discard_pile(self, new_pile):
        self.__discard_pile = new_pile
        np.resize(self.discard_pile, (52,))
        self.num_of_discard_cards = len(new_pile)

    @property
    def player_2(self):
        return self.__player_2

    @player_2.setter
    def player_2(self, new_player):
        self.__player_2 = <Player> new_player  # This is to fix problems converting between the normal

    @property
    def player_1(self):
        return self.__player_1

    @player_1.setter
    def player_1(self, new_player):
        self.__player_1 = <Player> new_player

    cdef int8_t pop_from_discard_pile(self):
        cdef int8_t to_return = self.__discard_pile[self.num_of_discard_cards - 1]
        self.num_of_discard_cards -= 1
        return to_return

    cdef void add_to_discard_pile(self, int8_t new_card):
        self.__discard_pile[self.num_of_discard_cards] = new_card
        self.num_of_discard_cards += 1
        return

    property SCORE_LIMIT:
        def __get__(self):
            return get_score_limit()
        def __set__(self, value):
            set_score_limit(value)

    property GIN_BONUS:
        def __get__(self):
            return get_gin_bonus()
        def __set__(self, value):
            set_gin_bonus(value)

    property BIG_GIN_BONUS:
        def __get__(self):
            return get_big_gin_bonus()
        def __set__(self, value):
            set_big_gin_bonus(value)

cdef int64_t _SCORE_LIMIT[1]
_SCORE_LIMIT[0] = 100
cdef int64_t get_score_limit(): return _SCORE_LIMIT[0]
cdef void set_score_limit(int64_t i): _SCORE_LIMIT[0] = i

cdef int64_t _GIN_BONUS[1]
_GIN_BONUS[0] = 25
cdef int64_t get_gin_bonus(): return _GIN_BONUS[0]
cdef void set_gin_bonus(int64_t i): _GIN_BONUS[0] = i

cdef int64_t _BIG_GIN_BONUS[1]
_BIG_GIN_BONUS[0] = 31
cdef int64_t get_big_gin_bonus(): return _BIG_GIN_BONUS[0]
cdef void set_big_gin_bonus(int64_t i): _BIG_GIN_BONUS[0] = i
