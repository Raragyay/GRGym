import numpy as np

from .card_enums cimport rank_symbols, rank_names, suit_symbols, suit_names
from libc.stdint cimport int64_t

cdef int a = 3

cdef class Hand:
    """
    A representation of the cards in a hand. Stored in a boolean array.
    Ordered by Suit, then by Rank. For example, Ace of Clubs is the 14th card (in position 13)
    """

    def __init__(self):
        self.__cards = np.zeros(52, dtype=np.bool)

    @property
    def cards(self):
        """
        Returns the numpy object of the memory view.
        :return:
        """
        return np.asarray(self.__cards, dtype=np.bool)

    @cards.setter
    def cards(self, new_arr):
        """
        Makes the internal cards property reference the array given.
        :param new_arr:
        :return:
        """
        self.__cards = new_arr

    cdef bint has_card(self, int64_t card_val):
        return self.__cards[card_val]

    cdef void add_card(self, int64_t card_val):
        self.__cards[card_val] = True

    cdef void remove_card(self, int64_t card_val):
        self.__cards[card_val] = False

    cdef np.ndarray card_list(self):
        return np.nonzero(self.cards)[0]

    def __str__(self):
        card_list = self.card_list()
        out = "Hand: \n"
        for card_val in card_list:
            card = divmod(card_val, 13)
            out += f"{rank_names()[card[1]]} of {suit_names()[card[0]]}\n"
        return out

    def __repr__(self):
        card_list = self.card_list()
        out = "Hand: "
        for card_val in card_list:
            out += f"{Hand.card_shorthand(card_val)} "
        return out

    @staticmethod
    cdef card_shorthand(int64_t card_val):
        card = divmod(card_val, 13)
        return f"{rank_symbols()[card[1]]}{suit_symbols()[card[0]]}"

    def __eq__(self, other):
        if isinstance(other, Hand):
            return np.array_equal(self.cards, other.cards)
        else:
            return False

    cdef Hand copy(self):
        cdef Hand new_hand = Hand()
        new_hand.cards = self.cards.copy()
        return new_hand
