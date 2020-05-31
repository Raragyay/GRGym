from .meld cimport Meld
from libc.stdint cimport int8_t
from .card_enums cimport rank_names, suit_names

cdef class Run(Meld):
    def __init__(self, int8_t start, int8_t end):
        # Maybe check that they are in same suit
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
        cdef:
            int8_t start_rank = self.start % 13
            int8_t end_rank = self.end % 13
        return f"Run from {rank_names()[start_rank]} of {suit_names()[self.suit]} to " \
               f"{rank_names()[end_rank]} of {suit_names()[self.suit]}"

    def __repr__(self):
        cdef:
            int8_t start_rank = self.start % 13
            int8_t end_rank = self.end % 13
        return f"Run | SUIT: {self.suit} | CARD_VALS: {self.start}-{self.end} | RANKS: {start_rank}-{end_rank}"

    def __eq__(self, other):
        cdef Run casted
        try:
            casted = other
            return self.start == casted.start and self.end == casted.end
        except TypeError as e:
            return False
