from libc.stdint cimport int64_t, uint8_t
import numpy as np
cimport numpy as np
cdef class Hand:
    cdef public uint8_t[:] __cards

    cpdef bint has_card(self, int64_t card_val)

    cpdef void add_card(self, int64_t card_val)
    cpdef void remove_card(self, int64_t card_val)
    cpdef np.ndarray card_list(self)
