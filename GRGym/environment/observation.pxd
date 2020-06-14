cimport numpy as np
from libc.stdint cimport int64_t, int8_t

cdef class Observation:
    cdef public PlayerID __player_id
    cdef public np.ndarray card_observations
    cdef public ActionPhase action_phase
    cdef public int64_t deck_size

cpdef enum PlayerID:
    ONE = 1
    TWO = 2

cpdef enum ActionPhase:
    DRAW, CALL_BEFORE_DISCARD, DISCARD, CALL_AFTER_DISCARD
