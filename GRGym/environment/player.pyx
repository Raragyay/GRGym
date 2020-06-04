import numpy as np

from .card_state cimport CardState
from .hand import Hand
from .hand cimport Hand
from libc.stdint cimport int8_t

cdef class Player:
    def __init__(self):
        self.hand = Hand()
        self.__card_states = np.zeros(52, dtype=np.int8)
        self.__score = 0

    @property
    def card_states(self):
        """
        Returns a copy of the internal card_state array.
        :return:
        """
        return np.asarray(self.__card_states, dtype=np.int8)

    @card_states.setter
    def card_states(self, np.ndarray new_arr):
        self.__card_states = new_arr

    @property
    def NO_CARD(self):
        return -1

    cdef void reset_hand(self):
        self.hand = Hand()
        self.__card_states = np.zeros(52, dtype=np.int8)

    cdef void add_card_from_deck(self, int8_t card_val):
        self.hand.add_card(card_val)
        self.__card_states[card_val] = CardState.MINE_FROM_DECK

    cdef void add_card_from_discard(self, int8_t card_val, int8_t new_top_of_discard):
        self.hand.add_card(card_val)
        self.__card_states[card_val] = CardState.MINE_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)

    cdef void report_opponent_drew_from_discard(self, int8_t card_val, int8_t new_top_of_discard):
        self.__card_states[card_val] = CardState.THEIRS_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)

    cdef void discard_card(self, int8_t card_to_discard, int8_t previous_top):
        self.hand.remove_card(card_to_discard)
        self.__card_states[card_to_discard] = CardState.DISCARD_MINE_TOP
        self.update_card_down(previous_top)

    cdef void report_opponent_discarded(self, int8_t card_discarded, int8_t previous_top):
        self.__card_states[card_discarded] = CardState.DISCARD_THEIRS_TOP
        self.update_card_down(previous_top)

    cdef void update_card_to_top(self, int8_t new_top_of_discard):
        if new_top_of_discard == self.NO_CARD:
            return
        if self.__card_states[new_top_of_discard] == CardState.DISCARD_THEIRS:
            self.__card_states[new_top_of_discard] = CardState.DISCARD_THEIRS_TOP
        elif self.__card_states[new_top_of_discard] == CardState.DISCARD_MINE:
            self.__card_states[new_top_of_discard] = CardState.DISCARD_MINE_TOP

    cdef void update_card_down(self, int8_t previous_top_of_discard):
        if previous_top_of_discard == self.NO_CARD:
            return
        if self.__card_states[previous_top_of_discard] == CardState.DISCARD_THEIRS_TOP:
            self.__card_states[previous_top_of_discard] = CardState.DISCARD_THEIRS
        elif self.__card_states[previous_top_of_discard] == CardState.DISCARD_MINE_TOP:
            self.__card_states[previous_top_of_discard] = CardState.DISCARD_MINE

    cdef bint has_card(self, int8_t card_val):
        return self.hand.has_card(card_val)

    cdef np.ndarray card_list(self):
        return self.hand.card_list()

    cdef np.ndarray hand_mask(self):
        return np.copy(self.hand.cards)

    cdef Player copy(self):
        cdef Player new_player = Player()
        new_player.hand = self.hand.copy()
        new_player.__card_states = self.__card_states.copy()
        new_player.score = self.score
        return new_player

    def __eq__(self, other):
        if isinstance(other, Player):
            casted = <Player> other
            return self.hand == casted.hand and np.array_equal(self.__card_states,
                                                               casted.__card_states) and self.score == casted.score
        else:
            return False

    def __repr__(self):
        return f'Player: {repr(self.hand)} | Score: {self.score} | Card State: {self.card_states}'

    def __str__(self):
        return f'Player:\n' \
               f'{self.hand}' \
               f'Score: {self.score}'

    property score:
        def __get__(self):
            return self.__score
        def __set__(self, value):
            self.__score = value
