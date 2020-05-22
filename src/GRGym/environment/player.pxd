from .hand cimport Hand
from libc.stdint cimport int8_t, int64_t
import numpy as np
cimport numpy as np
cdef class Player:
    cdef public:
        Hand hand
        int8_t[:] __card_states
        int64_t __score

    cpdef void reset_hand(self)
    cpdef void add_card_from_deck(self, int8_t card_val)
    cpdef void add_card_from_discard(self, int8_t card_val, int8_t new_top_of_discard)
    cpdef void report_opponent_drew_from_discard(self, int8_t card_val, int8_t new_top_of_discard)
    cpdef void discard_card(self, int8_t card_to_discard, int8_t previous_top)
    cpdef void report_opponent_discarded(self, int8_t card_discarded, int8_t previous_top)
    cpdef void update_card_to_top(self, int8_t new_top_of_discard)
    cpdef void update_card_down(self, int8_t previous_top_of_discard)
    cpdef bint has_card(self, int8_t card_val)
    cpdef np.ndarray card_list(self)
    cpdef np.ndarray hand_mask(self)

cpdef inline int8_t NO_CARD():
    return -1
