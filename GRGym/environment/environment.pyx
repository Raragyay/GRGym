import ctypes
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
from .observation cimport Observation, ActionPhase, PlayerID
from libc.stdint cimport int64_t

@cython.final
cdef class Environment:
    def __init__(self):
        self.__player_1 = Player()
        self.__player_2 = Player()
        self.__deck = np.arange(52, dtype=np.int8)
        self.__discard_pile = np.empty(52, dtype=np.int8)
        self.num_of_discard_cards = 0

        self.current_phase = ActionPhase.DRAW  # Either draw phase or discard phase
        self.current_player_id = PlayerID.ONE

    def reset(self) -> Observation:
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
        self.current_phase = ActionPhase.DRAW
        self.current_player_id = PlayerID.ONE
        return self.build_observations(self.player_1)

    def step(self, int64_t action) -> Tuple[np.ndarray, int]:
        # TODO FIRST CARD PRIVILEGE
        """
        Step through one interaction with the environment, given an action.

        :param action:
            An integer representing the action that is going to be taken.
            For each ActionPhase:
            DRAW: A boolean, with 0 representing drawing from discard, and 1 representing drawing from deck.
            CALL_BEFORE_DISCARD: A boolean, with 0 representing calling, and 1 representing not calling
            DISCARD: An integer ranging from [0, 52) representing the card ID that will be discarded
            CALL_AFTER_DISCARD: Same as CALL_BEFORE_DISCARD.

        :return:
            A tuple of results.
            The first element is the new observation.
            The second element is a boolean representing whether or not the game is finished.
            The third element is the reward.
        """
        cdef:
            bint done
            int64_t reward
            Player current_player = self.get_current_player()
        if self.current_phase == ActionPhase.DRAW:
            done, reward = self.run_draw(action, current_player)
        elif self.current_phase == ActionPhase.CALL_BEFORE_DISCARD or self.current_phase == \
                ActionPhase.CALL_AFTER_DISCARD:
            done, reward = self.run_call(action, current_player)
        elif self.current_phase == ActionPhase.DISCARD:
            done, reward = self.run_discard(action, current_player)
        else:
            raise ValueError(f"Current phase {self.current_phase} is not in the enum.")
        self.current_phase, self.current_player_id = self.advance_to_next_phase()
        # if self.draw_phase:
        #     action_result = self.run_draw(action, self.player_1)
        # else:  # Discard phase
        #     action_result = self.run_discard(action, self.player_1, is_player_1=True)
        # self.draw_phase = not self.draw_phase
        # if action_result != ActionResult.NO_CHANGE:
        #     self.reset_hand()
        # print(self)
        return self.build_observations(self.get_current_player()), done, reward

    def run_draw(self, int64_t wants_to_draw_from_deck, Player player) -> Tuple[bool, int]:
        # According to gin rummy rules, if there are only two cards left in the deck, it is a draw.
        if len(self.deck) == 2:
            return True, 0  # Game is done, player did not gain any points

        # Ensure a boolean was passed in
        assert wants_to_draw_from_deck == 0 or wants_to_draw_from_deck == 1
        if wants_to_draw_from_deck:
            self.draw_from_deck(player)
        else:
            self.draw_from_discard(player)
        return False, 0  # Game is not finished, player did not gain any points

    def run_call(self, int64_t wants_to_call, Player player)-> Tuple[bool, int]:
        cdef int64_t score_delta
        assert wants_to_call == 0 or wants_to_call == 1
        if self.current_phase == ActionPhase.CALL_BEFORE_DISCARD:
            if wants_to_call and Environment.is_gin(player):
                return True, self.get_opponent_deadwood(player) + self.BIG_GIN_BONUS
        elif self.current_phase == ActionPhase.CALL_AFTER_DISCARD:
            if wants_to_call:
                if Environment.is_gin(player):
                    return True, self.get_opponent_deadwood(player) + self.GIN_BONUS
                elif Environment.can_knock(player):
                    score_delta = self.try_to_knock(player)
                    return True, score_delta
        else:
            raise ValueError(f"Current ActionPhase {self.current_phase} is not a valid action phase for calling. ")

        return False, 0

    def run_discard(self, int64_t card_to_discard, Player player) -> Tuple[bool, int]:
        self.discard_card(player,card_to_discard)
        return False, 0
        # if Environment.wants_to_knock(action) and Environment.is_gin(player):
        #     return self.score_big_gin(player)
        # card_to_discard = self.get_card_to_discard(action, player)
        # self.discard_card(player, card_to_discard)
        # if Environment.wants_to_knock(action) and Environment.can_knock(player):
        #     if Environment.is_gin(player):
        #         return self.score_gin(player)
        #     else:
        #         score_delta = self.try_to_knock(player)
        #         if score_delta > 0:
        #             return self.update_score(player, score_delta)
        #         else:
        #             # Apply negative to return relative to current player
        #             return -self.update_score(self.opponents(player), -score_delta)
        # if not is_player_1:
        #     return 0  # Avoid player 2 recursing back into player 1, want to only recurse once
        # # Run player 2 draw and discard actions
        # opponent_draw_action = self.opponent_agent.act(self.build_observations(self.player_2))
        # self.run_draw(opponent_draw_action, self.player_2)
        # opponent_discard_action = self.opponent_agent.act(self.build_observations(self.player_2))
        # return self.run_discard(opponent_discard_action, self.player_2, False)

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

    def build_observations(self, Player player) -> Observation:
        """
        Returns the observation object for the given player.
        :param player:
        :return:
        """
        cdef Observation observation = Observation()
        observation.__player_id = self.current_player_id
        observation.card_observations = player.card_states
        observation.action_phase = self.current_phase
        observation.deck_size = len(self.deck)
        return observation

    def discard_card(self, Player player, card_to_discard: int):
        assert player.has_card(card_to_discard), f"Player does not have the card {card_to_discard}"
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

    cdef ActionResult score_big_gin(self, Player player):
        cdef int64_t score_delta = DeadwoodCounter(self.opponents(player).card_list()).deadwood() + get_big_gin_bonus()
        return self.update_score(player, score_delta)

    cdef ActionResult score_gin(self, Player player):
        cdef int64_t score_delta = DeadwoodCounter(self.opponents(player).card_list()).deadwood() + get_gin_bonus()
        return self.update_score(player, score_delta)

    cdef int64_t get_deadwood(self, Player player):
        return DeadwoodCounter(player.card_list()).deadwood()

    cdef int64_t get_opponent_deadwood(self, Player player):
        return self.get_deadwood(self.opponents(player))

    cdef ActionResult update_score(self, Player player, int64_t score_delta):
        player.score += score_delta
        if player.score >= self.SCORE_LIMIT:
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

    cdef Player get_current_player(self):
        if self.current_player_id == PlayerID.ONE:
            return self.player_1
        elif self.current_player_id == PlayerID.TWO:
            return self.player_2
        else:
            raise ValueError(f"Current player ID {self.current_player_id} is not part of the PlayerID Enum.")

    cdef Player opponents(self, Player player):
        """
        Opponents is implemented as a function instead of a dictionary to allow for swapping of players.
        :param player:
        :return: The opponent player.
        """
        if player is self.__player_1:
            return self.__player_2
        elif player is self.__player_2:
            return self.__player_1
        else:
            raise ValueError('This player is not currently in the environment.')

    cdef PlayerID next_player_id(self):
        if self.current_player_id == PlayerID.ONE:
            return PlayerID.TWO
        elif self.current_player_id == PlayerID.TWO:
            return PlayerID.ONE
        else:
            raise ValueError(f"Current player ID {self.current_player_id} is not part of the PlayerID Enum.")

    cdef (ActionPhase, PlayerID) advance_to_next_phase(self):
        if self.current_phase == ActionPhase.DRAW:
            return ActionPhase.CALL_BEFORE_DISCARD, self.current_player_id
        elif self.current_phase == ActionPhase.CALL_BEFORE_DISCARD:
            return ActionPhase.DISCARD, self.current_player_id
        elif self.current_phase == ActionPhase.DISCARD:
            return ActionPhase.CALL_AFTER_DISCARD, self.current_player_id
        else:
            return ActionPhase.DRAW, self.next_player_id()

    def __repr__(self):
        return f'{self.player_1.__repr__()}\n' \
               f'{self.player_2.__repr__()}\n' \
               f'Deck: {self.deck}\n' \
               f'Discard: {self.discard_pile}\n' \
               f'Current Phase:{self.current_phase}\n' \
               f'Current Player: {self.current_player_id}'

    cdef bint discard_pile_is_empty(self):
        return self.num_of_discard_cards == 0

    cdef void add_first_discard_card(self):
        self.add_to_discard_pile(self.__deck[0])
        self.opponents(self.player_1).report_opponent_discarded(self.__deck[0], self.player_1.NO_CARD)
        self.opponents(self.player_2).report_opponent_discarded(self.__deck[0], self.player_2.NO_CARD)
        self.__deck = self.__deck[1:]

    cdef int8_t pop_from_discard_pile(self):
        cdef int8_t to_return = self.__discard_pile[self.num_of_discard_cards - 1]
        self.num_of_discard_cards -= 1
        return to_return

    cdef void add_to_discard_pile(self, int8_t new_card):
        self.__discard_pile[self.num_of_discard_cards] = new_card
        self.num_of_discard_cards += 1
        return

    @staticmethod
    cdef void validate_card_array(np.ndarray card_array) except *:
        if card_array.ndim != 1:
            raise ValueError(f"The array provided has too many dimensions: {card_array.ndim}. Please reshape "
                             f"the array. \n"
                             f"Array provided: {card_array}")
        if card_array.size > 52:
            raise ValueError(f"The array provided is too large. Please provide an array with 52 elements or less. \n"
                             f"Array provided: {card_array}")
        if np.any(card_array < 0) or np.any(card_array >= 52):
            raise ValueError(f"The array provided has values that are not between [0,52). Please provide an array "
                             f"with correct card ids. \n"
                             f"Array provided: {card_array}")

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

    property player_1:
        def __get__(self):
            return self.__player_1
        def __set__(self, value):
            self.__player_1 = <Player> value

    property player_2:
        def __get__(self):
            return self.__player_2
        def __set__(self, value):
            self.__player_2 = <Player> value

    property deck:
        def __get__(self):
            return np.asarray(self.__deck, dtype=np.int8)
        def __set__(self, value):
            Environment.validate_card_array(value)
            self.__deck = value

    property discard_pile:
        """
        Property discard_pile: \n
        Setting the discard pile is done by value. 
        """
        def __get__(self):
            return np.asarray(self.__discard_pile[0:self.num_of_discard_cards], dtype=np.int8)
        def __set__(self, np.ndarray value):
            Environment.validate_card_array(value)
            self.__discard_pile = np.resize(value, (52,))
            self.num_of_discard_cards = value.size

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
