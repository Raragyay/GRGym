from .meld cimport Meld
from .card_enums cimport Rank, Suit
from libc.stdint cimport int8_t

cdef class Run(Meld):
    def __init__(self, int8_t start, int8_t end):
        self.start = start
        self.end = end
        self.suit = start // 13

    cdef set connectable_cards(self):
        cdef set result = set()
        if self.start % 13 != 0:  # not first card in suit
            result.add(self.start - 1)
        if (self.end + 1) % 13 != 0:  # not last card in suit
            result.add(self.end + 1)
        return result

    def __hash__(self):
        return hash(self.__repr__())

    def __str__(self):
        return f"Run from {Rank(self.start).name.title()} of {Suit(self.suit).name.title()} to " \
               f"{Rank(self.end).name.title()} of {Suit(self.suit).name.title()}"

    def __repr__(self):
        return f"R-{self.suit}-{self.start}-{self.end}"

    def __eq__(self, other):
        if isinstance(other, Run):
            casted = <Run> other
            return self.start == casted.start and self.end == casted.end
        else:
            return False
