cimport numpy as np
import numpy as np
from .types cimport INT64_T, INT32_T
ctypedef void (*ACTION_FUNC)(DeadwoodCounterRevised)
cdef class DeadwoodCounterRevised:
    """
    DeadwoodCounterDP(hand: np.ndarray)

    :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]

    Compiles the deadwood value for the given hand, the best set of melds, and the deadwood cards.
    """
    cdef np.ndarray hand
    cdef INT64_T[:] diamonds, clubs, hearts, spades
    cdef INT32_T cards_left_list[4]
    cdef INT64_T result[3]
    cdef ACTION_FUNC actions[3]
    cdef INT64_T dp[14*14*14*14*3]
    cdef INT64_T UNDEFINED
    #
    cpdef INT64_T deadwood(self)
    cdef void reset_cards_left_list(self)
    cpdef set remaining_cards(self)
    cpdef list melds(self)
    cdef void recurse(self)
    cdef bint in_dp(self)
    cdef Py_ssize_t cards_left_to_idx(self)
    cdef void set_dp(self, INT64_T deadwood, INT64_T cards_left, INT64_T melds)
    cdef void build_from_dp(self)
    cdef void build_result(self, INT64_T deadwood,INT64_T cards_left,INT64_T melds)
    cdef void try_to_drop_card(self)
    cdef void try_to_build_set(self)
    cdef void try_to_build_run(self)

    cdef INT64_T[:] suit_hands(DeadwoodCounterRevised self,Py_ssize_t suit)
    cdef INT32_T determine_max_run_length(DeadwoodCounterRevised self, INT32_T suit)

    @staticmethod
    cdef INT64_T c_deadwood_val(INT64_T card)
    cdef set bit_mask_to_array(DeadwoodCounterRevised self, INT64_T bit_mask)

    @staticmethod
    cdef list decode_meld_mask(INT64_T mask)
    @staticmethod
    cdef INT64_T add_set(INT64_T current_mask, INT64_T set_rank)
    @staticmethod
    cdef INT64_T add_run(INT64_T current_mask, INT64_T start, INT64_T end)
