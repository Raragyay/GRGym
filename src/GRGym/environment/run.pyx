from .meld cimport Meld

cdef class Run(Meld):
    def __init__(self, start: int, end: int):
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
        return hash(self.__str__())

    def __str__(self):
        return f"R-{self.suit}-{self.start}-{self.end}"

    def __repr__(self):
        return self.__str__()  # TODO MAKE NICER

    def __eq__(self, other):
        if isinstance(other, Run):
            casted = <Run> other
            return self.start == casted.start and self.end == casted.end
        else:
            return False
