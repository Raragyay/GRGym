import numpy as np

from card_state import CardState
from hand import Hand


class Player:
    def __init__(self):
        self.hand = Hand()
        self.card_states = np.zeros(52)
        self.score = 0

    def reset(self):
        self.hand = Hand()
        self.card_states = np.zeros(52)
        self.score = 0

    def add_card_from_deck(self, card_val: int):
        """
        Adds a card to the players deck and updates card_state observations.
        :param card_val:
        :return:
        """
        self.hand.add_card(card_val)
        self.card_states[card_val] = CardState.MINE_FROM_DECK

    def add_card_from_discard(self, card_val: int, new_top_of_discard: int):
        self.hand.add_card(card_val)
        self.card_states[card_val] = CardState.MINE_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)
        # TODO logging if none of them are the case

    def report_opponent_drew_from_discard(self, card_val: int, new_top_of_discard: int):
        self.card_states[card_val] = CardState.THEIRS_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)

    def update_card_to_top(self, new_top_of_discard: int):
        if self.card_states[new_top_of_discard] == CardState.DISCARD_THEIRS:
            self.card_states[new_top_of_discard] = CardState.DISCARD_THEIRS_TOP
        elif self.card_states[new_top_of_discard] == CardState.DISCARD_MINE:
            self.card_states[new_top_of_discard] = CardState.DISCARD_MINE_TOP

    def has_card(self, card_val: int):
        return self.hand.has_card(card_val)

    def card_list(self):
        return self.hand.card_list()

    def hand_mask(self):
        return np.copy(self.hand.cards)

    def discard_card(self, card_to_discard: int, previous_top: int):
        self.hand.remove_card(card_to_discard)
        self.card_states[card_to_discard] = CardState.DISCARD_MINE_TOP
        self.update_card_down(previous_top)

    def update_card_down(self, previous_top_of_discard: int):
        if self.card_states[previous_top_of_discard] == CardState.DISCARD_THEIRS_TOP:
            self.card_states[previous_top_of_discard] = CardState.DISCARD_THEIRS
        elif self.card_states[previous_top_of_discard] == CardState.DISCARD_MINE_TOP:
            self.card_states[previous_top_of_discard] = CardState.DISCARD_MINE

    def report_opponent_discarded(self, card_discarded, previous_top):
        self.card_states[card_discarded] = CardState.DISCARD_THEIRS_TOP
        self.update_card_down(previous_top)
