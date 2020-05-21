import numpy as np

from .card_enums import Rank, Suit
from src.GRGym.core.types cimport INT64_T

suit_symbols = ['D', 'C', 'H', 'S']
rank_symbols = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
cdef class Hand:
    """
    A representation of the cards in a hand. Stored in a boolean array.
    Ordered by Suit, then by Rank. For example, Ace of Clubs is the 14th card (in position 13)
    """

    def __init__(self):
        self.__cards = np.zeros(52, dtype=np.bool)

    @property
    def cards(self):
        return np.asarray(self.__cards, dtype=np.bool)

    @cards.setter
    def cards(self, new_arr):
        self.__cards = new_arr

    cpdef bint has_card(self, INT64_T card_val):
        return self.__cards[card_val]

    cpdef void add_card(self, INT64_T card_val):
        self.__cards[card_val] = True

    cpdef void remove_card(self, INT64_T card_val):
        self.__cards[card_val] = False

    cpdef np.ndarray card_list(self):
        return np.nonzero(self.cards)[0]

    def __str__(self):
        card_list = self.card_list()
        out = "Hand: \n"
        for card_val in card_list:
            card = divmod(card_val, 13)
            out += f"{Rank(card[1]).name.title()} of {Suit(card[0]).name.title()}\n"
        return out

    def __repr__(self):
        card_list = self.card_list()
        out = "Hand: "
        for card_val in card_list:
            out += f"{self.card_shorthand(card_val)} "
        return out

    @classmethod
    def card_shorthand(cls, card_val: int):
        card = divmod(card_val, 13)
        return f"{rank_symbols[card[1]]}{suit_symbols[card[0]]}"

    def __eq__(self, other):
        return isinstance(other, Hand) and np.array_equal(self.cards, other.cards)

    def __deepcopy__(self, memodict={}):
        new_hand = Hand()
        new_hand.cards = self.cards.copy()
        return new_hand
