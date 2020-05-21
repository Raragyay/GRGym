from .hand cimport Hand
from src.GRGym.core.types cimport INT8_T, INT64_T
import numpy as np
cimport numpy as np
cdef class Player:
    cdef public:
        Hand hand
        INT8_T[:] __card_states
        INT64_T __score

    cpdef void reset_hand(self)
    cpdef void add_card_from_deck(self, INT8_T card_val)
    cpdef void add_card_from_discard(self, INT8_T card_val, INT8_T new_top_of_discard)
    cpdef void report_opponent_drew_from_discard(self, INT8_T card_val, INT8_T new_top_of_discard)
    cpdef void discard_card(self, INT8_T card_to_discard, INT8_T previous_top)
    cpdef void report_opponent_discarded(self, INT8_T card_discarded, INT8_T previous_top)
    cpdef void update_card_to_top(self, INT8_T new_top_of_discard)
    cpdef void update_card_down(self, INT8_T previous_top_of_discard)
    cpdef bint has_card(self, INT8_T card_val)
    cpdef np.ndarray card_list(self)
    cpdef np.ndarray hand_mask(self)

cpdef inline INT8_T NO_CARD():
    return -1
