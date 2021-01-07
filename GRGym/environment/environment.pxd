from .observation cimport Observation
from .player cimport Player
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

    cpdef step(self, int64_t action)
    cdef (bint, int64_t) run_draw(self, int64_t wants_to_draw_from_deck, Player player)
    cdef (bint, int64_t) run_call(self, int64_t wants_to_call, Player player)
    cdef (bint, int64_t) run_discard(self, int64_t card_to_discard, Player player)

    cdef Observation build_observations(self, Player player)

    cdef void draw_from_deck(self, Player player, int64_t num_of_cards = *) except *
    cdef void draw_from_discard(self, Player player) except *
    cdef void discard_card(self, Player player, card_to_discard: int) except *
    cdef int64_t try_to_knock(self, Player player)

    @staticmethod
    cdef int64_t get_deadwood(Player player)
    cdef int64_t get_opponent_deadwood(self, Player player)

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
