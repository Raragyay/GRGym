cimport cython
import numpy as np
cimport numpy as np
from libc.string cimport memset
from libc.limits cimport INT_MAX
from .run cimport Run
from .set cimport Set
from libc.stdint cimport int32_t, int64_t

@cython.final
cdef class DeadwoodCounter:
    """
    DeadwoodCounterDP(hand: np.ndarray)

    :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]

    Compiles the deadwood value for the given hand, the best set of melds, and the deadwood cards.
    """

    def __init__(self, np.ndarray[int64_t, ndim=1] hand):
        """
        Hand must be in suit then rank form, ascending order

        :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]
        """
        self.hand = hand
        self.diamonds = self.hand[self.hand < 13]
        self.clubs = self.hand[np.logical_and(self.hand >= 13, self.hand < 26)]
        self.hearts = self.hand[np.logical_and(self.hand >= 26, self.hand < 39)]
        self.spades = self.hand[self.hand >= 39]

        self.cards_left_list = [0, 0, 0, 0]
        self.result = [0, 0, 0]

        self.actions = [self.try_to_build_set, self.try_to_build_run, self.try_to_drop_card]
        self.UNDEFINED = 0x3f3f3f3f3f3f3f3f
        memset(self.dp, 0x3f, 14 * 14 * 14 * 14 * 3 * sizeof(int64_t))

    cdef int64_t deadwood(self):
        self.reset_cards_left_list()
        self.recurse()
        return self.result[0]

    cdef set remaining_cards(self):
        self.reset_cards_left_list()
        self.recurse()
        return DeadwoodCounter.bit_mask_to_array(self.result[1])

    cdef set melds(self):
        self.reset_cards_left_list()
        self.recurse()
        return DeadwoodCounter.decode_meld_mask(self.result[2])

    cdef void reset_cards_left_list(self):
        cdef Py_ssize_t i
        for i in range(4):
            self.cards_left_list[i] = len(self.suit_hands(i))

    cdef void recurse(self):
        """
        TODO DOCUMENT
        :return:
        """
        # noinspection PyTypeChecker
        if self.in_dp():
            self.build_from_dp()
            return
        if self.cards_left_list[0] == self.cards_left_list[1] == self.cards_left_list[2] == self.cards_left_list[
            3] == 0:
            # all cards used
            self.set_dp(0LL, 0LL, 0LL)
            self.build_from_dp()
            return

        cdef:
            int64_t lowest_deadwood = INT_MAX
            int64_t lowest_deadwood_remaining_cards = 0LL
            int64_t lowest_deadwood_melds = 0LL
            int64_t prospective_deadwood, prospective_remaining_cards, prospective_melds

        for action in self.actions:
            action(self)
            prospective_deadwood = self.result[0]
            prospective_remaining_cards = self.result[1]
            prospective_melds = self.result[2]
            if prospective_deadwood < lowest_deadwood:
                lowest_deadwood = prospective_deadwood
                lowest_deadwood_remaining_cards = prospective_remaining_cards
                lowest_deadwood_melds = prospective_melds
            if lowest_deadwood == 0:
                self.set_dp(lowest_deadwood, lowest_deadwood_remaining_cards, lowest_deadwood_melds)
                self.build_from_dp()
                return

        self.set_dp(lowest_deadwood, lowest_deadwood_remaining_cards, lowest_deadwood_melds)
        self.build_from_dp()
        return

    cdef void set_dp(self, int64_t deadwood, int64_t cards_left, int64_t melds):
        self.dp[self.cards_left_to_idx() + 0] = deadwood
        self.dp[self.cards_left_to_idx() + 1] = cards_left
        self.dp[self.cards_left_to_idx() + 2] = melds

    cdef Py_ssize_t cards_left_to_idx(self):
        return self.cards_left_list[0] * 14 * 14 * 14 * 3 + self.cards_left_list[1] * 14 * 14 * 3 + \
               self.cards_left_list[2] * 14 * 3 + self.cards_left_list[3] * 3

    cdef void build_from_dp(self):
        self.build_result(
            self.dp[self.cards_left_to_idx() + 0],
            self.dp[self.cards_left_to_idx() + 1],
            self.dp[self.cards_left_to_idx() + 2])

    cdef bint in_dp(self):
        return self.dp[self.cards_left_to_idx() + 0] != self.UNDEFINED

    cdef void build_result(self, int64_t deadwood, int64_t cards_left, int64_t melds):
        self.result[0] = deadwood
        self.result[1] = cards_left
        self.result[2] = melds

    cdef void try_to_drop_card(self):
        cdef:
            int64_t lowest_deadwood = INT_MAX
            int64_t lowest_deadwood_remaining_cards = 0LL
            int64_t lowest_deadwood_melds = 0LL
            Py_ssize_t suit
            int32_t cards_left
            int64_t ignored_card, ignored_card_deadwood
            int64_t prospective_deadwood, prospective_remaining_cards, prospective_melds

        for suit in range(4):
            cards_left = self.cards_left_list[suit]
            if cards_left == 0:
                continue
            ignored_card = self.suit_hands(suit)[self.cards_left_list[suit] - 1]
            ignored_card_deadwood = DeadwoodCounter.deadwood_val(ignored_card)
            self.cards_left_list[suit] -= 1  # Ignore card

            self.recurse()
            prospective_deadwood = self.result[0]
            prospective_remaining_cards = self.result[1]
            prospective_melds = self.result[2]

            if prospective_deadwood + ignored_card_deadwood < lowest_deadwood:
                lowest_deadwood = prospective_deadwood + ignored_card_deadwood
                lowest_deadwood_remaining_cards = DeadwoodCounter.encode_card(prospective_remaining_cards, ignored_card)
                lowest_deadwood_melds = prospective_melds
            self.cards_left_list[suit] += 1  # restore card
        self.build_result(lowest_deadwood, lowest_deadwood_remaining_cards, lowest_deadwood_melds)
        return

    cdef void try_to_build_set(self):
        # Get rank of last card in each suit
        cdef:
            Py_ssize_t i, j
            int64_t last_ranks[4]
            int32_t counter_arr[18]
            int64_t max_freq_rank = 0
            int32_t frequency = 0
            int64_t last_rank
            int64_t suits_with_set_rank[4]
        memset(counter_arr, 0, 18 * sizeof(int32_t))

        # Find most frequent rank and its frequency

        for i in range(4):
            if self.cards_left_list[i] > 0:
                last_rank = self.suit_hands(i)[self.cards_left_list[i] - 1] % 13
            else:
                last_rank = -i - 1
            last_ranks[i] = last_rank
            counter_arr[last_rank + 5] += 1

        for i in range(18):
            if counter_arr[i] > frequency:
                max_freq_rank = i - 5
                frequency = counter_arr[i]

        if frequency < 3:  # Cannot form set
            self.build_result(INT_MAX, 0LL, 0LL)
            return

        # Determine all suits with set rank
        j = 0
        for i in range(4):
            if last_ranks[i] == max_freq_rank:
                suits_with_set_rank[j] = i
                j += 1

        for i in range(frequency):  # Use all available cards
            self.cards_left_list[suits_with_set_rank[i]] -= 1

        if frequency == 3:  # Can only form 1 set
            self.recurse()
            deadwood = self.result[0]
            remaining_cards = self.result[1]
            melds = self.result[2]
            melds = DeadwoodCounter.add_set(melds, max_freq_rank)
            self.build_result(deadwood, remaining_cards, melds)
        else:  # all suits have same rank
            self.recurse()
            lowest_deadwood = self.result[0]
            lowest_remaining_cards = self.result[1]
            lowest_melds = self.result[2]

            lowest_melds = DeadwoodCounter.add_set(lowest_melds, max_freq_rank)

            for excluded_suit in range(4):
                self.cards_left_list[excluded_suit] += 1  # Restore card

                self.recurse()
                prospective_deadwood = self.result[0]
                prospective_remaining_cards = self.result[1]
                prospective_melds = self.result[2]

                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
                    lowest_remaining_cards = prospective_remaining_cards
                    lowest_melds = DeadwoodCounter.add_set(prospective_melds, max_freq_rank)
                self.cards_left_list[excluded_suit] -= 1  # Use card

            self.build_result(lowest_deadwood, lowest_remaining_cards, lowest_melds)

        for i in range(frequency):  # Restore all cards
            self.cards_left_list[suits_with_set_rank[i]] += 1
        return

    cdef void try_to_build_run(self):  #
        cdef:
            int64_t lowest_deadwood = INT_MAX
            int64_t lowest_remaining_cards = 0LL
            int64_t lowest_melds = 0LL
            int32_t max_run_length
            int64_t run_start_card, run_end_card

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

                self.recurse()
                prospective_deadwood = self.result[0]
                prospective_remaining_cards = self.result[1]
                prospective_melds = self.result[2]

                if prospective_deadwood < lowest_deadwood:
                    lowest_deadwood = prospective_deadwood
                    lowest_remaining_cards = prospective_remaining_cards
                    run_start_card = self.suit_hands(suit)[self.cards_left_list[suit]]
                    lowest_melds = DeadwoodCounter.add_run(prospective_melds, run_start_card, run_end_card)
            self.cards_left_list[suit] += max_run_length

        self.build_result(lowest_deadwood, lowest_remaining_cards, lowest_melds)
        return

    cdef int64_t[:] suit_hands(self, Py_ssize_t suit):
        if suit == 0:
            return self.diamonds
        elif suit == 1:
            return self.clubs
        elif suit == 2:
            return self.hearts
        elif suit == 3:
            return self.spades

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef int32_t determine_max_run_length(self, int32_t suit):
        cdef int64_t[:] suit_hand = self.suit_hands(suit)
        cdef int32_t max_run_length = self.cards_left_list[suit] # This will always be >=3
        cdef int64_t prev_rank = suit_hand[max_run_length - 1] % 13
        cdef int32_t run_length = 1

        while run_length < max_run_length:
            if prev_rank - suit_hand[max_run_length - 1 - run_length] % 13 == 1:  # Is consecutive
                prev_rank -= 1
                run_length += 1
            else:
                break
        return run_length

    @staticmethod
    cdef inline int64_t deadwood_val(int64_t card):
        cdef int64_t rank = card % 13
        if rank >= 9:
            return 10
        else:
            return rank + 1  # zero-indexed

    @staticmethod
    cdef int64_t encode_card(int64_t prospective_remaining_cards, int64_t ignored_card):
        return prospective_remaining_cards | (1LL << ignored_card)

    @staticmethod
    cdef set bit_mask_to_array(int64_t bit_mask):
        return {bit for bit in range(52) if (bit_mask & (1LL << bit)) != 0}

    """
    Masks are 64-bit integers. 
    Starting from the LSB, each group of 12 bits represents an encoding for the runs in that suit. 
    The reason why we don't represent a suit with 13 bits is because 13 * 4 + 13 = 65, 
    which cannot be fit inside a 64 bit integer. 
    Instead, we can check if a run has been started to conclude whether or not a run ends at the King card.    
    A run from Ace of Diamonds to 4 of Diamonds would be encoded at bits 0 and 3.
    A run from Jack of Diamonds to King of Diamonds would only be encoded at bit 10. 
    King of Diamonds (12) would not be encoded, but since the run was not closed, we can assume that it ends at King.
    The following 13 bits are used to denote whether or not a set exists at the given rank. 
    For example, a set of aces would be encoded at bit 48.
    """

    @staticmethod
    cdef int64_t add_run(int64_t current_mask, int64_t start, int64_t end):
        current_mask |= (1LL << ((start // 13 * 12) + start % 13))
        if(end - 12) % 13 != 0:  # Is not a king
            current_mask |= (1LL << ((end // 13 * 12) + end % 13))
        return current_mask

    @staticmethod
    cdef int64_t add_set(int64_t current_mask, int64_t set_rank):
        return current_mask | (1LL << (48 + set_rank))

    @staticmethod
    cdef set decode_meld_mask(int64_t mask):
        cdef:
            bint run_started = False
            Py_ssize_t i, j
            int64_t starting_card
            set melds = set()

        for i in range(4):
            for j in range(12):
                if mask & 1 == 1:  # If the current card is the start or end of a run
                    if not run_started:
                        starting_card = i * 13 + j
                        run_started = True
                    else:
                        run_started = False
                        melds.add(Run(starting_card, i * 13 + j))
                mask >>= 1
            if run_started: # If there is a run that hasn't been finished yet
                run_started = False
                # Since j represents the queen card, we increment by 1 to represent the king card
                melds.add(Run(starting_card, i * 13 + j + 1))

        for i in range(13):
            if mask & 1 == 1:
                melds.add(Set(i))
            mask >>= 1

        return melds
