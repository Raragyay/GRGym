import numpy as np

from card_state cimport CardState
from hand import Hand
from hand cimport Hand
from types cimport INT8_T

cdef class Player:
    def __init__(self):
        self.hand = Hand()
        self.__card_states = np.zeros(52, dtype=np.int8)
        self.__score = 0

    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, new_score):
        self.__score = new_score

    @property
    def card_states(self):
        return np.asarray(self.__card_states, dtype=np.int8)

    @card_states.setter
    def card_states(self, new_arr):
        self.__card_states = new_arr

    cpdef void reset_hand(self):
        self.hand = Hand()
        self.__card_states = np.zeros(52, dtype=np.int8)

    cpdef void add_card_from_deck(self, INT8_T card_val):
        self.hand.add_card(card_val)
        self.__card_states[card_val] = CardState.MINE_FROM_DECK

    cpdef void add_card_from_discard(self, INT8_T card_val, INT8_T new_top_of_discard):
        self.hand.add_card(card_val)
        self.__card_states[card_val] = CardState.MINE_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)

    cpdef void report_opponent_drew_from_discard(self, INT8_T card_val, INT8_T new_top_of_discard):
        self.__card_states[card_val] = CardState.THEIRS_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)

    cpdef void discard_card(self, INT8_T card_to_discard, INT8_T previous_top):
        self.hand.remove_card(card_to_discard)
        self.__card_states[card_to_discard] = CardState.DISCARD_MINE_TOP
        self.update_card_down(previous_top)

    cpdef void report_opponent_discarded(self, INT8_T card_discarded, INT8_T previous_top):
        self.__card_states[card_discarded] = CardState.DISCARD_THEIRS_TOP
        self.update_card_down(previous_top)

    cpdef void update_card_to_top(self, INT8_T new_top_of_discard):
        if new_top_of_discard == NO_CARD():
            return
        if self.__card_states[new_top_of_discard] == CardState.DISCARD_THEIRS:
            self.__card_states[new_top_of_discard] = CardState.DISCARD_THEIRS_TOP
        elif self.__card_states[new_top_of_discard] == CardState.DISCARD_MINE:
            self.__card_states[new_top_of_discard] = CardState.DISCARD_MINE_TOP

    cpdef void update_card_down(self, INT8_T previous_top_of_discard):
        if previous_top_of_discard == NO_CARD():
            return
        if self.__card_states[previous_top_of_discard] == CardState.DISCARD_THEIRS_TOP:
            self.__card_states[previous_top_of_discard] = CardState.DISCARD_THEIRS
        elif self.__card_states[previous_top_of_discard] == CardState.DISCARD_MINE_TOP:
            self.__card_states[previous_top_of_discard] = CardState.DISCARD_MINE

    cpdef bint has_card(self, INT8_T card_val):
        return self.hand.has_card(card_val)

    cpdef np.ndarray card_list(self):
        return self.hand.card_list()

    cpdef np.ndarray hand_mask(self):
        return np.copy(self.hand.cards)

    def __eq__(self, other):
        if isinstance(other, Player):
            return self.hand == other.hand and np.array_equal(self.card_states, other.card_states) and self.score == \
                   other.score
        else:
            return False

    def __repr__(self):
        return f'Player: {self.hand.__repr__()} | {self.score} | {self.card_states}'

    def __str__(self):
        return f'Player:\n' \
               f'Hand: {self.hand}\n' \
               f'Score: {self.score}'

    def __deepcopy__(self, memodict={}):
        new_player = Player()
        new_player.hand = self.hand.__deepcopy__()
        new_player.card_states = self.card_states.copy()
        new_player.score = self.score
        return new_player
