cimport numpy as np
import numpy as np

ctypedef np.int_t INT32_T
ctypedef np.longlong_t INT64_T

cdef class DeadwoodCounterRevised:
    """
    DeadwoodCounterDP(hand: np.ndarray)

    :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]

    Compiles the deadwood value for the given hand, the best set of melds, and the deadwood cards.
    """
    cdef np.ndarray hand
    cdef INT64_T[:] diamonds, clubs, hearts, spades
    # cdef int[:] suit_hands[4]
    cdef dict deadwood_cards_dp, melds_dp, dp
    cdef INT32_T cards_left_list[4]
    #
    # cpdef int deadwood(self)
    cdef void reset_cards_left_list(self)
    # cpdef set remaining_cards(self)
    # cpdef tuple melds(self)

    cdef INT64_T[:] suit_hands(DeadwoodCounterRevised self,INT32_T suit)
    cdef INT32_T determine_max_run_length(DeadwoodCounterRevised self, INT32_T suit)
    # cdef (INT32_T,INT64_T,INT64_T) try_to_build_run(self)

    @staticmethod
    cdef INT32_T c_deadwood_val(INT32_T card)
    cdef set bit_mask_to_array(DeadwoodCounterRevised self, INT64_T bit_mask)

    @staticmethod
    cdef list decode_meld_mask(INT64_T mask)
    @staticmethod
    cdef INT64_T add_set(INT64_T current_mask,INT64_T set_rank)
    @staticmethod
    cdef INT64_T add_run(INT64_T current_mask,INT64_T start,INT64_T end)
