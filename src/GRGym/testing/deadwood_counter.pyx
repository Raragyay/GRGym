import numpy as np
cimport numpy as np
from src.GRGym.core.types cimport INT32_T, INT64_T
from src.GRGym.environment.deadwood_counter cimport DeadwoodCounter as ProdDeadwoodCounter

cdef class DeadwoodCounter:
    def __init__(self, np.ndarray[INT64_T, ndim=1] hand):
        self.__internal_counter = ProdDeadwoodCounter(hand)

    def deadwood(self) -> INT64_T:
        return self.__internal_counter.deadwood()

    def remaining_cards(self) -> set:
        return self.__internal_counter.remaining_cards()

    def melds(self) -> set:
        return self.__internal_counter.melds()

    @staticmethod
    def deadwood_val(INT32_T card) -> int:
        return ProdDeadwoodCounter.c_deadwood_val(card)
