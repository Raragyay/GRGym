import sys
import typing
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from deadwood.deadwood_counter import DeadwoodCounter
from meld.meld import Meld
from meld.run import Run
from meld.set import Set


class DeadwoodCounterDP(DeadwoodCounter):
    """
    DeadwoodCounterDP(hand: np.ndarray)

    :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]

    Compiles the deadwood value for the given hand, the best set of melds, and the deadwood cards.
    """

    def __init__(self, hand: np.ndarray):
        """
        Hand must be in suit then rank form, ascending order

        :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]
        """
        super().__init__(hand)
        self.diamonds: np.ndarray = self.hand[self.hand < 13]
        self.clubs: np.ndarray = self.hand[np.logical_and(self.hand >= 13, self.hand < 26)]
        self.hearts: np.ndarray = self.hand[np.logical_and(self.hand >= 26, self.hand < 39)]
        self.spades: np.ndarray = self.hand[self.hand >= 39]

        self.suit_hands: Tuple = (self.diamonds, self.clubs, self.hearts, self.spades)

        self.deadwood_cards_dp: [Tuple[int, int, int, int], int] = dict()
        self.melds_dp: [Tuple[int, int, int, int], Tuple[Meld, ...]] = dict()
        self.dp: Dict[Tuple[int, int, int, int], int] = dict()
        self.cards_left_list: List[int] = []

    def deadwood(self) -> int:
        self.cards_left_list = [len(suit_hand) for suit_hand in self.suit_hands]
        return self.recurse()[0]

    def remaining_cards(self) -> typing.Set[int]:
        self.cards_left_list = [len(suit_hand) for suit_hand in self.suit_hands]
        return self.bit_mask_to_array(self.recurse()[1])

    def melds(self) -> Tuple[Meld, ...]:
        self.cards_left_list = [len(suit_hand) for suit_hand in self.suit_hands]
        return self.recurse()[2]

    def recurse(self) -> Tuple[int, int, Tuple[Meld, ...]]:
        """
        TODO DOCUMENT
        :return:
        """
        # noinspection PyTypeChecker
        cards_left_tuple: Tuple[int, int, int, int] = tuple(self.cards_left_list)
        if cards_left_tuple in self.dp:
            return self.dp_retrieve(cards_left_tuple)
        if sum(cards_left_tuple) == 0:  # all cards used
            self.dp[cards_left_tuple] = 0
            self.deadwood_cards_dp[cards_left_tuple] = 0
            self.melds_dp[cards_left_tuple] = tuple()
            return self.dp_retrieve(cards_left_tuple)

        lowest_deadwood = sys.maxsize
        lowest_deadwood_remaining_cards: int = 0
        lowest_deadwood_melds: Tuple[Meld, ...] = tuple()

        for action in (self.try_to_build_set, self.try_to_build_run, self.try_to_drop_card):
            prospective_deadwood, prospective_remaining_cards, prospective_melds = action()
            if prospective_deadwood < lowest_deadwood:
                lowest_deadwood = prospective_deadwood
                lowest_deadwood_remaining_cards = prospective_remaining_cards
                lowest_deadwood_melds = prospective_melds
            if lowest_deadwood == 0:
                self.dp[cards_left_tuple] = lowest_deadwood
                self.deadwood_cards_dp[cards_left_tuple] = lowest_deadwood_remaining_cards
                self.melds_dp[cards_left_tuple] = lowest_deadwood_melds
                return self.dp_retrieve(cards_left_tuple)

        self.dp[cards_left_tuple] = lowest_deadwood
        self.deadwood_cards_dp[cards_left_tuple] = lowest_deadwood_remaining_cards
        self.melds_dp[cards_left_tuple] = lowest_deadwood_melds
        return self.dp_retrieve(cards_left_tuple)

    def dp_retrieve(self, cards_left_tuple) -> Tuple[int, int, Tuple[Meld, ...]]:
        return self.dp[cards_left_tuple], self.deadwood_cards_dp[cards_left_tuple], self.melds_dp[cards_left_tuple]

    def try_to_drop_card(self) -> Tuple[int, int, Tuple[Meld, ...]]:
        lowest_deadwood = sys.maxsize
        lowest_deadwood_remaining_cards: int = 0
        lowest_deadwood_melds: Tuple[Meld, ...] = tuple()
        for suit, cards_left in enumerate(self.cards_left_list):
            if cards_left == 0:
                continue
            ignored_card = self.suit_hands[suit][self.cards_left_list[suit] - 1]
            ignored_card_deadwood = self.deadwood_val(ignored_card)
            self.cards_left_list[suit] -= 1  # Ignore card
            prospective_deadwood, prospective_remaining_cards, prospective_melds = self.recurse()
            if prospective_deadwood + ignored_card_deadwood < lowest_deadwood:
                lowest_deadwood = prospective_deadwood + ignored_card_deadwood
                lowest_deadwood_remaining_cards = prospective_remaining_cards | (1 << ignored_card)
                lowest_deadwood_melds = prospective_melds
            self.cards_left_list[suit] += 1  # restore card
        return lowest_deadwood, lowest_deadwood_remaining_cards, lowest_deadwood_melds

    def try_to_build_set(self) -> Tuple[int, int, Tuple[Meld, ...]]:
        # Get rank of last card in each suit
        last_ranks = [suit_hand[self.cards_left_list[suit] - 1] % 13 if
                      self.cards_left_list[suit] > 0 else -suit - 1 for suit, suit_hand in enumerate(self.suit_hands)]
        last_ranks_count = Counter(last_ranks)
        # Find rank that can be a set
        set_rank: Tuple[int, int] = last_ranks_count.most_common(1)[0]  # rank_val, frequency
        if set_rank[1] < 3:  # Cannot form set
            return sys.maxsize, 0, tuple()
        suits_with_set_rank = [idx for idx in range(4) if last_ranks[idx] == set_rank[0]]
        if len(suits_with_set_rank) == 3:  # Can only form 1 set
            for suit in suits_with_set_rank:  # Use card
                self.cards_left_list[suit] -= 1
            deadwood, remaining_cards, melds = self.recurse()
            melds = melds + (Set(rank=set_rank[0]),)
            for suit in suits_with_set_rank:  # Restore card
                self.cards_left_list[suit] += 1
            return deadwood, remaining_cards, melds
        else:  # all suits have same rank
            # use all suits
            for suit in range(4):
                self.cards_left_list[suit] -= 1
            lowest_deadwood, lowest_remaining_cards, lowest_melds = self.recurse()
            lowest_melds = lowest_melds + (Set(rank=set_rank[0]),)

            for excluded_suit in range(4):
                self.cards_left_list[excluded_suit] += 1  # Restore card
                prospective_deadwood, prospective_remaining_cards, prospective_melds = self.recurse()
                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
                    lowest_remaining_cards = prospective_remaining_cards
                    lowest_melds = prospective_melds + (Set(rank=set_rank[0]),)
                self.cards_left_list[excluded_suit] -= 1  # Use card
            for suit in range(4):
                self.cards_left_list[suit] += 1  # Restore all cards
            return lowest_deadwood, lowest_remaining_cards, lowest_melds

    def try_to_build_run(self) -> Tuple[int, int, Tuple[Meld, ...]]:
        lowest_deadwood: int = sys.maxsize
        lowest_remaining_cards: int = 0
        lowest_melds: Tuple[Meld, ...] = tuple()
        for suit in range(4):
            if self.cards_left_list[suit] < 3:  # Not enough cards to build run
                continue
            max_run_length = self.determine_max_run_length(suit)
            run_end_card = self.suit_hands[suit][self.cards_left_list[suit] - 1]
            if max_run_length < 3:
                continue
            self.cards_left_list[suit] -= 2
            for i in range(max_run_length - 2):
                self.cards_left_list[suit] -= 1
                prospective_deadwood, prospective_remaining_cards, prospective_melds = self.recurse()
                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
                    lowest_remaining_cards = prospective_remaining_cards
                    run_start_card = self.suit_hands[suit][self.cards_left_list[suit]]
                    lowest_melds = prospective_melds + (Run(run_start_card, run_end_card),)
            self.cards_left_list[suit] += max_run_length
        return lowest_deadwood, lowest_remaining_cards, lowest_melds

    def determine_max_run_length(self, suit: int) -> int:
        suit_hand = self.suit_hands[suit]
        prev_rank = suit_hand[self.cards_left_list[suit] - 1] % 13
        max_run_length = 1
        while max_run_length < self.cards_left_list[suit]:
            if prev_rank - suit_hand[self.cards_left_list[suit] - 1 - max_run_length] % 13 == 1:  # Is consecutive
                prev_rank -= 1
                max_run_length += 1
            else:
                break
        return max_run_length

    @staticmethod
    def deadwood_val(card: int) -> int:
        rank = card % 13
        if rank >= 9:
            return 10
        else:
            return rank + 1  # zero-indexed

    @staticmethod
    def bit_mask_to_array(bit_mask):
        return {bit for bit in range(52) if (bit_mask & (1 << bit)) != 0}
