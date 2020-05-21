from .meld cimport Meld

cdef class Set(Meld):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    cpdef set connectable_cards(self):
        return {suit * 13 + self.rank for suit in range(4)}

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return f"S{self.rank}"

    def __repr__(self):
        return self.__str__()  # TODO MAKE NICER

    def __eq__(self, other):
        try:
            casted = <Set> other
            return self.rank == casted.rank
        finally:
            pass
