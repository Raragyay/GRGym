import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class DeadwoodCounter:

    def __init__(self, hand: np.ndarray):
        """
        Hand must be in suit then rank form, ascending order
        :param hand:
        """
        self.hand: np.ndarray((52,)) = hand
        self.diamonds: np.ndarray = self.hand[self.hand < 13]
        self.clubs: np.ndarray = self.hand[np.logical_and(self.hand >= 13, self.hand < 26)]
        self.hearts: np.ndarray = self.hand[np.logical_and(self.hand >= 26, self.hand < 39)]
        self.spades: np.ndarray = self.hand[self.hand >= 39]

        self.suit_hands: Tuple = (self.diamonds, self.clubs, self.hearts, self.spades)

        self.dp: Dict[Tuple[int], int] = dict()

    def deadwood(self) -> int:
        return self.recurse([len(suit_hand) for suit_hand in self.suit_hands])

    def recurse(self, cards_left_list: List[int]) -> int:
        """
        TODO DOCUMENT
        :param cards_left_list:
        :return:
        """
        cards_left_tuple = tuple(cards_left_list)
        if cards_left_tuple in self.dp:
            return self.dp[cards_left_tuple]
        if sum(cards_left_tuple) == 0:  # all cards used
            self.dp[cards_left_tuple] = 0
            return 0

        lowest_deadwood = sys.maxsize

        drop_card_lowest = self.try_to_drop_card(cards_left_list)
        if drop_card_lowest < lowest_deadwood:
            lowest_deadwood = drop_card_lowest

        build_set_lowest = self.try_to_build_set(cards_left_list)
        if build_set_lowest < lowest_deadwood:
            lowest_deadwood = build_set_lowest

        build_run_lowest = self.try_to_build_run(cards_left_list)
        if build_run_lowest < lowest_deadwood:
            lowest_deadwood = build_run_lowest

        self.dp[cards_left_tuple] = lowest_deadwood

        return lowest_deadwood

    def try_to_drop_card(self, cards_left_list) -> int:
        lowest_deadwood = sys.maxsize
        for suit, cards_left in enumerate(cards_left_list):
            if cards_left == 0:
                continue
            ignored_card_deadwood = self.deadwood_val(self.suit_hands[suit][cards_left_list[suit] - 1])
            cards_left_list[suit] -= 1  # Ignore card
            propsective_deadwood = self.recurse(cards_left_list) + ignored_card_deadwood
            if propsective_deadwood < lowest_deadwood:
                lowest_deadwood = propsective_deadwood
            cards_left_list[suit] += 1  # restore card
        return lowest_deadwood

    def try_to_build_set(self, cards_left_list) -> int:
        # Get rank of last card in each suit
        last_ranks = [suit_hand[cards_left_list[suit] - 1] % 13 if
                      cards_left_list[suit] > 0 else -suit - 1 for suit, suit_hand in enumerate(self.suit_hands)]
        last_ranks_count = Counter(last_ranks)
        # Find rank that can be a set
        set_rank: tuple = last_ranks_count.most_common(1)[0]
        if set_rank[1] < 3:  # Cannot form set
            return sys.maxsize
        suits_with_set_rank = [idx for idx in range(4) if last_ranks[idx] == set_rank[0]]
        if len(suits_with_set_rank) == 3:  # Can only form 1 set
            for suit in suits_with_set_rank:  # Use card
                cards_left_list[suit] -= 1
            deadwood = self.recurse(cards_left_list)
            for suit in suits_with_set_rank:  # Restore card
                cards_left_list[suit] += 1
            return deadwood
        else:  # all suits have same rank
            # use all suits
            for suit in range(4):
                cards_left_list[suit] -= 1
            lowest_deadwood = self.recurse(cards_left_list)

            for excluded_suit in range(4):
                cards_left_list[excluded_suit] += 1  # Restore card
                prospective_deadwood = self.recurse(cards_left_list)
                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
                cards_left_list[excluded_suit] -= 1  # Use card
            for suit in range(4):
                cards_left_list[suit] += 1
            return lowest_deadwood

    def try_to_build_run(self, cards_left_list) -> int:
        lowest_deadwood = sys.maxsize
        for suit in range(4):
            suit_hand = self.suit_hands[suit]
            if cards_left_list[suit] < 3:  # Not enough cards to build run
                continue
            prev_rank = suit_hand[cards_left_list[suit] - 1] % 13
            max_run_length = 1
            while max_run_length < cards_left_list[suit]:
                if prev_rank - suit_hand[cards_left_list[suit] - 1 - max_run_length] % 13 == 1:  # Is consecutive
                    prev_rank -= 1
                    max_run_length += 1
                else:
                    break
            if max_run_length < 3:
                continue
            cards_left_list[suit] -= 2
            for i in range(max_run_length - 2):
                cards_left_list[suit] -= 1
                prospective_deadwood = self.recurse(cards_left_list)
                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
            cards_left_list[suit] += max_run_length
        return lowest_deadwood

    @staticmethod
    def deadwood_val(card: int) -> int:
        rank = card % 13
        if rank >= 9:
            return 10
        else:
            pass
        return rank + 1  # zero-indexed


if __name__ == '__main__':
    arr = np.arange(2)
    print(arr)
    c = DeadwoodCounter(arr)
    print(c.deadwood())
