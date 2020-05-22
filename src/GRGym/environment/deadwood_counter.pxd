cimport numpy as np
import numpy as np
from libc.stdint cimport int64_t, int32_t
ctypedef void (*ACTION_FUNC)(DeadwoodCounter)
cdef class DeadwoodCounter:
    """
    DeadwoodCounterDP(hand: np.ndarray)

    :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]

    Compiles the deadwood value for the given hand, the best set of melds, and the deadwood cards.
    """
    cdef np.ndarray hand
    cdef int64_t[:] diamonds, clubs, hearts, spades
    cdef int32_t cards_left_list[4]
    cdef int64_t result[3]
    cdef ACTION_FUNC actions[3]
    cdef int64_t dp[14 * 14 * 14 * 14 * 3]
    cdef int64_t UNDEFINED
    #
    cdef int64_t deadwood(self)
    cdef set remaining_cards(self)
    cdef set melds(self)
    cdef void reset_cards_left_list(self)

    cdef void recurse(self)

    cdef bint in_dp(self)
    cdef Py_ssize_t cards_left_to_idx(self)
    cdef void set_dp(self, int64_t deadwood, int64_t cards_left, int64_t melds)
    cdef void build_from_dp(self)
    cdef void build_result(self, int64_t deadwood,int64_t cards_left,int64_t melds)

    cdef void try_to_drop_card(self)
    cdef void try_to_build_set(self)
    cdef void try_to_build_run(self)

    cdef int64_t[:] suit_hands(self,Py_ssize_t suit)
    cdef int32_t determine_max_run_length(self, int32_t suit)

    @staticmethod
    cdef int64_t deadwood_val(int64_t card)

    @staticmethod
    cdef int64_t encode_card(int64_t prospective_remaining_cards, int64_t ignored_card)
    @staticmethod
    cdef set bit_mask_to_array(int64_t bit_mask)

    @staticmethod
    cdef set decode_meld_mask(int64_t mask)
    @staticmethod
    cdef int64_t add_set(int64_t current_mask, int64_t set_rank)
    @staticmethod
    cdef int64_t add_run(int64_t current_mask, int64_t start, int64_t end)
