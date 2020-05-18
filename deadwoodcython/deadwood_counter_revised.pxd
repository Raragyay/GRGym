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
    cpdef INT64_T[:] suit_hands(DeadwoodCounterRevised self,INT32_T suit)
    cdef INT32_T determine_max_run_length(DeadwoodCounterRevised self, INT32_T suit)
    cpdef INT32_T deadwood_val(DeadwoodCounterRevised self, INT32_T card)
