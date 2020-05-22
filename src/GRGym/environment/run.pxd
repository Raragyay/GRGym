from .meld cimport Meld
from libc.stdint cimport int32_t

cdef class Run(Meld):
    cdef:
        int32_t start, end, suit
    cdef set connectable_cards(self)
