cimport numpy as np
import numpy as np

ctypedef np.int_t INT32_T
ctypedef np.longlong_t LL_T

cdef class DeadwoodCounterRevised:
    """
    DeadwoodCounterDP(hand: np.ndarray)

    :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]

    Compiles the deadwood value for the given hand, the best set of melds, and the deadwood cards.
    """
    cdef np.ndarray hand, diamonds, clubs, hearts, spades
    cdef list suit_hands
    cdef dict deadwood_cards_dp, melds_dp, dp
    cdef list cards_left_list
    #
    # cpdef int deadwood(self)
    # cdef void reset_cards_left_list(self)
    # cpdef set remaining_cards(self)
    # cpdef tuple melds(self)
    cpdef INT32_T deadwood_val(DeadwoodCounterRevised self, INT32_T card)
