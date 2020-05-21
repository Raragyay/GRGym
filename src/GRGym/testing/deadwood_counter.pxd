from src.GRGym.environment.deadwood_counter cimport DeadwoodCounter as ProdDeadwoodCounter

cdef class DeadwoodCounter:
    cdef:
        ProdDeadwoodCounter __internal_counter
