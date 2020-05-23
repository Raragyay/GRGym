from libc.stdint cimport int64_t, uint8_t
import numpy as np
cimport numpy as np


cdef class Hand:
    cdef uint8_t[:] __cards #uint8_t is the C version of np.bool

    cdef bint has_card(self, int64_t card_val)
    cdef void add_card(self, int64_t card_val)
    cdef void remove_card(self, int64_t card_val)
    cdef np.ndarray card_list(self)

    @staticmethod
    cdef card_shorthand(int64_t card_val)
    cdef copy(self)
