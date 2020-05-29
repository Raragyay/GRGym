import cython

from .meld cimport Meld
from libc.stdint cimport int32_t
from .card_enums cimport Rank

@cython.final
cdef class Set(Meld):
    def __init__(self, int32_t rank):
        self.rank = rank

    cdef set connectable_cards(self):
        return {suit * 13 + self.rank for suit in range(4)}

    def __hash__(self):
        return hash(self.__repr__())

    def __str__(self):
        return f"Set of rank {Rank(self.rank).name.title()}"

    def __repr__(self):
        return f"S{self.rank}"

    def __eq__(self, other):
        try:
            casted = <Set> other
            return self.rank == casted.rank
        except TypeError:
            return False
