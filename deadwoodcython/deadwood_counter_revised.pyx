# cython: profile=True

import sys
import typing
from collections import Counter
from typing import Tuple

cimport cython
import numpy as np
cimport numpy as np

from meld.meld import Meld
from meld.run import Run
from meld.set import Set

INT32 = np.int
cdef class DeadwoodCounterRevised:
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
        self.hand = hand
        self.diamonds = self.hand[self.hand < 13]
        self.clubs = self.hand[np.logical_and(self.hand >= 13, self.hand < 26)]
        self.hearts = self.hand[np.logical_and(self.hand >= 26, self.hand < 39)]
        self.spades = self.hand[self.hand >= 39]

        # self.suit_hands = [self.diamonds, self.clubs, self.hearts, self.spades]

        self.deadwood_cards_dp = dict()
        self.melds_dp = dict()
        self.dp = dict()
        self.cards_left_list = [0,0,0,0]

    def deadwood(self) -> int:
        self.reset_cards_left_list()
        return self.recurse()[0]

    cdef void reset_cards_left_list(self):
        cdef Py_ssize_t i
        for i in range(4):
            self.cards_left_list[i] = len(self.suit_hands(i))

    def remaining_cards(self) -> typing.Set[int]:
        self.reset_cards_left_list()
        return self.bit_mask_to_array(self.recurse()[1])

    def melds(self) -> Tuple[Meld, ...]:
        self.reset_cards_left_list()
        return DeadwoodCounterRevised.decode_meld_mask(self.recurse()[2])

    def recurse(self) -> Tuple[int, int, int]:
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
            self.melds_dp[cards_left_tuple] = 0LL
            return self.dp_retrieve(cards_left_tuple)

        lowest_deadwood = sys.maxsize
        lowest_deadwood_remaining_cards: int = 0
        lowest_deadwood_melds: int = 0LL

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
        cdef INT64_T lowest_deadwood_melds = 0LL
        for suit, cards_left in enumerate(self.cards_left_list):
            if cards_left == 0:
                continue
            ignored_card = self.suit_hands(suit)[self.cards_left_list[suit] - 1]
            ignored_card_deadwood = DeadwoodCounterRevised.c_deadwood_val(ignored_card)
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
        cdef Py_ssize_t i
        cdef INT32_T last_ranks[4]
        for i in range(4):
             if self.cards_left_list[i] > 0:
                 last_ranks[i]=self.suit_hands(i)[self.cards_left_list[i] - 1] % 13
             else:
                 last_ranks[i]=-i-1
        # last_ranks = [suit_hand[self.cards_left_list[suit] - 1] % 13 if
        #               self.cards_left_list[suit] > 0 else -suit - 1 for suit, suit_hand in enumerate(self.suit_hands)]
        last_ranks_count = Counter(last_ranks)
        # Find rank that can be a set
        cdef (INT32_T,INT32_T) set_rank = last_ranks_count.most_common(1)[0]  # rank_val, frequency
        if set_rank[1] < 3:  # Cannot form set
            return sys.maxsize, 0, tuple()
        suits_with_set_rank = [idx for idx in range(4) if last_ranks[idx] == set_rank[0]]
        if len(suits_with_set_rank) == 3:  # Can only form 1 set
            for suit in suits_with_set_rank:  # Use card
                self.cards_left_list[suit] -= 1
            deadwood, remaining_cards, melds = self.recurse()
            melds = DeadwoodCounterRevised.add_set(melds,set_rank[0])
            for suit in suits_with_set_rank:  # Restore card
                self.cards_left_list[suit] += 1
            return deadwood, remaining_cards, melds
        else:  # all suits have same rank
            # use all suits
            for suit in range(4):
                self.cards_left_list[suit] -= 1
            lowest_deadwood, lowest_remaining_cards, lowest_melds = self.recurse()
            lowest_melds = DeadwoodCounterRevised.add_set(lowest_melds,set_rank[0])

            for excluded_suit in range(4):
                self.cards_left_list[excluded_suit] += 1  # Restore card
                prospective_deadwood, prospective_remaining_cards, prospective_melds = self.recurse()
                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
                    lowest_remaining_cards = prospective_remaining_cards
                    lowest_melds = DeadwoodCounterRevised.add_set(prospective_melds, set_rank[0])
                self.cards_left_list[excluded_suit] -= 1  # Use card
            for suit in range(4):
                self.cards_left_list[suit] += 1  # Restore all cards
            return lowest_deadwood, lowest_remaining_cards, lowest_melds

    def try_to_build_run(self) -> Tuple[int, int, Tuple[Meld, ...]]:
        lowest_deadwood: int = sys.maxsize
        lowest_remaining_cards: int = 0
        cdef INT64_T lowest_melds = 0LL
        for suit in range(4):
            if self.cards_left_list[suit] < 3:  # Not enough cards to build run
                continue
            max_run_length = self.determine_max_run_length(suit)
            run_end_card = self.suit_hands(suit)[self.cards_left_list[suit] - 1]
            if max_run_length < 3:
                continue
            self.cards_left_list[suit] -= 2
            for i in range(max_run_length - 2):
                self.cards_left_list[suit] -= 1
                prospective_deadwood, prospective_remaining_cards, prospective_melds = self.recurse()
                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
                    lowest_remaining_cards = prospective_remaining_cards
                    run_start_card = self.suit_hands(suit)[self.cards_left_list[suit]]
                    lowest_melds = DeadwoodCounterRevised.add_run(prospective_melds, run_start_card, run_end_card)
            self.cards_left_list[suit] += max_run_length
        return lowest_deadwood, lowest_remaining_cards, lowest_melds

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef INT32_T determine_max_run_length(DeadwoodCounterRevised self, INT32_T suit):
        cdef INT64_T[:] suit_hand = self.suit_hands(suit)
        cdef INT32_T max_run_length = self.cards_left_list[suit] # This will always be >=3
        cdef INT64_T prev_rank = suit_hand[max_run_length - 1] % 13
        cdef INT32_T run_length = 1

        while run_length < max_run_length:
            if prev_rank - suit_hand[max_run_length - 1 - run_length] % 13 == 1:  # Is consecutive
                prev_rank -= 1
                run_length += 1
            else:
                break
        return run_length

    @staticmethod
    def deadwood_val(INT32_T card):
        return DeadwoodCounterRevised.c_deadwood_val(card)

    @staticmethod
    cdef INT32_T c_deadwood_val(INT32_T card):
        cdef INT32_T rank = card % 13
        if rank >= 9:
            return 10
        else:
            return rank + 1  # zero-indexed

    cdef set bit_mask_to_array(DeadwoodCounterRevised self, INT64_T bit_mask):
        return {bit for bit in range(52) if (bit_mask & (1LL << bit)) != 0}

    """
    Masks are in the form [13*SET][Q-A RUN]
    For example, a set of aces would be encoded at bit 48
    A run from Ace of diamonds to 4 of diamonds would be encoded at bits 0 and 3
    A run from Jack of diamonds to King of diamonds would only be encoded at bit 10. King of diamonds (12) would not 
    be encoded
    """
    @staticmethod
    cdef INT64_T add_run(INT64_T current_mask,INT64_T start,INT64_T end):
        current_mask|=(1LL<<((start//13*12)+start%13))
        if(end-12)%13!=0:  # Is not a king
            current_mask|=(1LL<<((end//13*12)+end%13))
        return current_mask

    @staticmethod
    cdef INT64_T add_set(INT64_T current_mask,INT64_T set_rank):
        return current_mask|(1LL<<(48+set_rank))

    @staticmethod
    cdef list decode_meld_mask(INT64_T mask):
        cdef bint run_started=False
        cdef Py_ssize_t i,j
        cdef INT32_T starting_card
        cdef list melds=[]
        # for i in range(4):
        #     for j in range(12):
        #         if counter_mask&1==1: #If run is set
        #             if not run_started:
        #                 arr_size_counter+=1
        #             else:
        #                 run_started=True
        #         counter_mask>>=1
        #     if run_started:
        #         run_started=False
        #         arr_size_counter+=1
        # for i in range(13):
        #     if counter_mask&1==1:
        #         arr_size_counter+=1
        #     counter_mask>>=1

        for i in range(4):
            for j in range(12):
                if mask&1==1:  # If run is set
                    if not run_started:
                        starting_card=i*13+j
                        run_started=True
                    else:
                        run_started=False
                        melds.append(Run(starting_card,i*13+j))
                mask>>=1
            if run_started:
                run_started=False
                melds.append(Run(starting_card,i*13+j))
        for i in range(13):
            if mask&1==1:
                melds.append(Set(i))
            mask>>=1
        return melds

    cdef INT64_T[:] suit_hands(DeadwoodCounterRevised self,INT32_T suit):
        if suit==0:
            return self.diamonds
        elif suit==1:
            return self.clubs
        elif suit==2:
            return self.hearts
        elif suit==3:
            return self.spades
