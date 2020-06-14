from .player cimport Player
from .player import Player
from libc.stdint cimport int8_t, int64_t
from .action_result cimport ActionResult
from .observation cimport ActionPhase, PlayerID
cimport numpy as np

cdef class Environment:
    cdef:
        Player __player_1, __player_2

        int8_t[:] __deck
        int8_t[:] __discard_pile
        ActionPhase current_phase
        PlayerID current_player_id
        Py_ssize_t num_of_discard_cards

    cdef ActionResult score_big_gin(self, Player player)
    cdef ActionResult score_gin(self, Player player)
    cdef int64_t get_deadwood(self, Player player)
    cdef int64_t get_opponent_deadwood(self, Player player)
    cdef ActionResult update_score(self, Player player, int64_t score_delta)

    @staticmethod
    cdef bint wants_to_draw_from_deck(double[:] action)
    @staticmethod
    cdef bint wants_to_knock(double[:] action)
    @staticmethod
    cdef bint is_gin(Player player)
    @staticmethod
    cdef bint can_knock(Player player)

    cdef Player get_current_player(self)
    cdef Player opponents(self, Player player)
    cdef PlayerID next_player_id(self)
    cdef (ActionPhase, PlayerID) advance_to_next_phase(self)

    cdef bint discard_pile_is_empty(self)
    cdef void add_first_discard_card(self)
    cdef int8_t pop_from_discard_pile(self)
    cdef void add_to_discard_pile(self, int8_t new_card)
    @staticmethod
    cdef void validate_card_array(np.ndarray card_array) except *
