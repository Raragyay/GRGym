from .player cimport Player
from .player import Player
from src.GRGym.core.types cimport INT8_T, BOOL_T
cdef class CythonEnvironment:
    cdef:
        Player __player_1, __player_2
        object opponent_agent
    cdef:
        INT8_T[:] __deck
        INT8_T[:] __discard_pile
        BOOL_T draw_phase
        Py_ssize_t num_of_discard_cards

    @staticmethod
    cdef bint c_is_gin(Player player)

    @staticmethod
    cdef bint c_can_knock(Player player)

    cdef Player opponents(self, Player player)
    cdef bint discard_pile_is_empty(self)
    cdef void add_first_discard_card(self)
    cdef INT8_T pop_from_discard_pile(self)
    cdef void add_to_discard_pile(self, INT8_T new_card)
