from meld cimport Meld
from types cimport INT32_T

cdef class Run(Meld):
    cdef public:
        INT32_T start, end, suit
    cpdef set connectable_cards(self)
