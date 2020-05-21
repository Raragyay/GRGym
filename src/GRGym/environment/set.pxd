from .meld cimport Meld
from src.GRGym.core.types cimport INT32_T

cdef class Set(Meld):
    cdef public:
        INT32_T rank
    cpdef set connectable_cards(self)
