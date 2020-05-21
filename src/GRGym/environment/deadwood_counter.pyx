cimport cython
import numpy as np
cimport numpy as np
from libc.string cimport memset
from libc.limits cimport INT_MAX
from .run cimport Run
from .set cimport Set
from .types cimport INT32_T, INT64_T

@cython.final
cdef class DeadwoodCounter:
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

        self.cards_left_list = [0, 0, 0, 0]
        self.result = [0, 0, 0]

        self.actions = [self.try_to_build_set, self.try_to_build_run, self.try_to_drop_card]
        self.UNDEFINED = 0x3f3f3f3f3f3f3f3f
        memset(self.dp, 0x3f, 14 * 14 * 14 * 14 * 3 * sizeof(INT64_T))

    cpdef INT64_T deadwood(self):
        self.reset_cards_left_list()
        self.recurse()
        return self.result[0]

    cdef void reset_cards_left_list(self):
        cdef Py_ssize_t i
        for i in range(4):
            self.cards_left_list[i] = len(self.suit_hands(i))

    cpdef set remaining_cards(self):
        self.reset_cards_left_list()
        self.recurse()
        return self.bit_mask_to_array(self.result[1])

    cpdef list melds(self):
        self.reset_cards_left_list()
        self.recurse()
        return DeadwoodCounter.decode_meld_mask(self.result[2])

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
            INT64_T lowest_deadwood = INT_MAX
            INT64_T lowest_deadwood_remaining_cards = 0LL
            INT64_T lowest_deadwood_melds = 0LL
            INT64_T prospective_deadwood, prospective_remaining_cards, prospective_melds

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

    cdef void set_dp(self, INT64_T deadwood, INT64_T cards_left, INT64_T melds):
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

    cdef void build_result(self, INT64_T deadwood, INT64_T cards_left, INT64_T melds):
        self.result[0] = deadwood
        self.result[1] = cards_left
        self.result[2] = melds

    cdef void try_to_drop_card(self):
        cdef:
            INT64_T lowest_deadwood = INT_MAX
            INT64_T lowest_deadwood_remaining_cards = 0LL
            INT64_T lowest_deadwood_melds = 0LL
            Py_ssize_t suit
            INT32_T cards_left
            INT64_T ignored_card, ignored_card_deadwood
            INT64_T prospective_deadwood, prospective_remaining_cards, prospective_melds

        for suit in range(4):
            cards_left = self.cards_left_list[suit]
            if cards_left == 0:
                continue
            ignored_card = self.suit_hands(suit)[self.cards_left_list[suit] - 1]
            ignored_card_deadwood = DeadwoodCounter.c_deadwood_val(ignored_card)
            self.cards_left_list[suit] -= 1  # Ignore card

            self.recurse()
            prospective_deadwood = self.result[0]
            prospective_remaining_cards = self.result[1]
            prospective_melds = self.result[2]

            if prospective_deadwood + ignored_card_deadwood < lowest_deadwood:
                lowest_deadwood = prospective_deadwood + ignored_card_deadwood
                lowest_deadwood_remaining_cards = prospective_remaining_cards | (1LL << ignored_card)
                lowest_deadwood_melds = prospective_melds
            self.cards_left_list[suit] += 1  # restore card
        self.build_result(lowest_deadwood, lowest_deadwood_remaining_cards, lowest_deadwood_melds)
        return

    cdef void try_to_build_set(self):
        # Get rank of last card in each suit
        cdef:
            Py_ssize_t i, j
            INT64_T last_ranks[4]
            INT32_T counter_arr[18]
            INT64_T max_freq_rank = 0
            INT32_T frequency = 0
            INT64_T last_rank
            INT64_T suits_with_set_rank[4]
        memset(counter_arr, 0, 18 * sizeof(INT32_T))

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
            INT64_T lowest_deadwood = INT_MAX
            INT64_T lowest_remaining_cards = 0LL
            INT64_T lowest_melds = 0LL
            INT32_T max_run_length
            INT64_T run_start_card, run_end_card

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

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef INT32_T determine_max_run_length(DeadwoodCounter self, INT32_T suit):
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
    def deadwood_val(INT32_T card) -> int:
        return DeadwoodCounter.c_deadwood_val(card)

    @staticmethod
    cdef inline INT64_T c_deadwood_val(INT64_T card):
        cdef INT64_T rank = card % 13
        if rank >= 9:
            return 10
        else:
            return rank + 1  # zero-indexed

    cdef set bit_mask_to_array(self, INT64_T bit_mask):
        return {bit for bit in range(52) if (bit_mask & (1LL << bit)) != 0}

    """
    Masks are in the form [13*SET][Q-A RUN]
    For example, a set of aces would be encoded at bit 48
    A run from Ace of diamonds to 4 of diamonds would be encoded at bits 0 and 3
    A run from Jack of diamonds to King of diamonds would only be encoded at bit 10. King of diamonds (12) would not 
    be encoded
    """
    @staticmethod
    cdef INT64_T add_run(INT64_T current_mask, INT64_T start, INT64_T end):
        current_mask |= (1LL << ((start // 13 * 12) + start % 13))
        if(end - 12) % 13 != 0:  # Is not a king
            current_mask |= (1LL << ((end // 13 * 12) + end % 13))
        return current_mask

    @staticmethod
    cdef INT64_T add_set(INT64_T current_mask, INT64_T set_rank):
        return current_mask | (1LL << (48 + set_rank))

    @staticmethod
    cdef list decode_meld_mask(INT64_T mask):
        cdef:
            bint run_started = False
            Py_ssize_t i, j
            INT64_T starting_card
            list melds = []

        for i in range(4):
            for j in range(12):
                if mask & 1 == 1:  # If run is set
                    if not run_started:
                        starting_card = i * 13 + j
                        run_started = True
                    else:
                        run_started = False
                        melds.append(Run(starting_card, i * 13 + j))
                mask >>= 1
            if run_started:
                run_started = False
                melds.append(Run(starting_card, i * 13 + j))
        for i in range(13):
            if mask & 1 == 1:
                melds.append(Set(i))
            mask >>= 1
        return melds

    cdef INT64_T[:] suit_hands(DeadwoodCounter self, Py_ssize_t suit):
        if suit == 0:
            return self.diamonds
        elif suit == 1:
            return self.clubs
        elif suit == 2:
            return self.hearts
        elif suit == 3:
            return self.spades
