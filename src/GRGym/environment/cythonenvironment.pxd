from .player cimport Player
from .player import Player
from libc.stdint cimport int8_t, int64_t
cdef class CythonEnvironment:
    cdef:
        Player __player_1, __player_2
        object opponent_agent
    cdef:
        int8_t[:] __deck
        int8_t[:] __discard_pile
        bint draw_phase
        Py_ssize_t num_of_discard_cards

    # cdef int64_t try_to_knock(self, Player player)

    @staticmethod
    cdef bint c_is_gin(Player player)

    @staticmethod
    cdef bint c_can_knock(Player player)

    cdef Player opponents(self, Player player)
    cdef bint discard_pile_is_empty(self)
    cdef void add_first_discard_card(self)
    cdef int8_t pop_from_discard_pile(self)
    cdef void add_to_discard_pile(self, int8_t new_card)
