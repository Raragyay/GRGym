import numpy as np

from card_enums import Rank, Suit


class Hand:
    """
    A representation of the cards in a hand. Stored in a boolean array.
    Ordered by Suit, then by Rank. For example, Ace of Clubs is the 14th card (in position 13)
    """
    suit_symbols = ['♦', '♣', '♥', '♠']
    rank_symbols = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']

    def __init__(self):
        self.cards: np.ndarray((52,), np.bool) = np.zeros(52, np.bool)

    def has_card(self, card_val) -> bool:
        return self.cards[card_val]

    def add_card(self, card_val: int):
        self.cards[card_val] = True

    def remove_card(self, card_val: int):
        self.cards[card_val] = False

    def card_list(self):
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
            card = divmod(card_val, 13)
            out += f"{self.rank_symbols[card[1]]}{self.suit_symbols[card[0]]} "
        return out
