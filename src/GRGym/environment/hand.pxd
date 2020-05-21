from .types cimport INT64_T, BOOL_T
cimport numpy as np
import numpy as np

cdef class Hand:
    cdef public BOOL_T[:] __cards

    cpdef bint has_card(self, INT64_T card_val)

    cpdef void add_card(self, INT64_T card_val)
    cpdef void remove_card(self, INT64_T card_val)
    cpdef np.ndarray card_list(self)
