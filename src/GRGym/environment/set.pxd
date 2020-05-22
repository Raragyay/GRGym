from .meld cimport Meld
from libc.stdint cimport int32_t

cdef class Set(Meld):
    cdef:
        int32_t rank

    cdef set connectable_cards(self)
