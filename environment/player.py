import numpy as np

from environment.card_state import CardState
from environment.hand import Hand


class Player:
    def __init__(self):
        self.hand = Hand()
        self.card_states = np.zeros(52)
        self.__score = 0

    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, new_score):
        self.__score = new_score

    def reset(self):
        self.hand = Hand()
        self.card_states = np.zeros(52)

    def add_card_from_deck(self, card_val: int):
        self.hand.add_card(card_val)
        self.card_states[card_val] = CardState.MINE_FROM_DECK

    def add_card_from_discard(self, card_val: int, new_top_of_discard: int):
        self.hand.add_card(card_val)
        self.card_states[card_val] = CardState.MINE_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)

    def report_opponent_drew_from_discard(self, card_val: int, new_top_of_discard: int):
        self.card_states[card_val] = CardState.THEIRS_FROM_DISCARD
        self.update_card_to_top(new_top_of_discard)

    def discard_card(self, card_to_discard: int, previous_top: int):
        self.hand.remove_card(card_to_discard)
        self.card_states[card_to_discard] = CardState.DISCARD_MINE_TOP
        self.update_card_down(previous_top)

    def report_opponent_discarded(self, card_discarded: int, previous_top: int):
        self.card_states[card_discarded] = CardState.DISCARD_THEIRS_TOP
        self.update_card_down(previous_top)

    def update_card_to_top(self, new_top_of_discard: int):
        if new_top_of_discard is None:
            return
        if self.card_states[new_top_of_discard] == CardState.DISCARD_THEIRS:
            self.card_states[new_top_of_discard] = CardState.DISCARD_THEIRS_TOP
        elif self.card_states[new_top_of_discard] == CardState.DISCARD_MINE:
            self.card_states[new_top_of_discard] = CardState.DISCARD_MINE_TOP

    def update_card_down(self, previous_top_of_discard: int):
        if previous_top_of_discard is None:
            return
        if self.card_states[previous_top_of_discard] == CardState.DISCARD_THEIRS_TOP:
            self.card_states[previous_top_of_discard] = CardState.DISCARD_THEIRS
        elif self.card_states[previous_top_of_discard] == CardState.DISCARD_MINE_TOP:
            self.card_states[previous_top_of_discard] = CardState.DISCARD_MINE

    def has_card(self, card_val: int) -> bool:
        return self.hand.has_card(card_val)

    def card_list(self) -> np.ndarray:
        return self.hand.card_list()

    def hand_mask(self) -> np.ndarray:
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
