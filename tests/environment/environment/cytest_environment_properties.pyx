include "cython_wrapper.pxi"
import numpy as np
cimport numpy as np
import pytest
from GRGym.environment.player cimport Player
from GRGym.environment.environment cimport Environment

@cython_wrap
def test_score_limit(Environment test_env, test_agent):
    cdef Environment other_test_env = Environment(test_agent)
    assert other_test_env.SCORE_LIMIT == test_env.SCORE_LIMIT
    test_env.SCORE_LIMIT += 1
    assert other_test_env.SCORE_LIMIT == test_env.SCORE_LIMIT
    test_env.SCORE_LIMIT = 500
    assert other_test_env.SCORE_LIMIT == test_env.SCORE_LIMIT
    test_env.SCORE_LIMIT = 23
    assert other_test_env.SCORE_LIMIT == test_env.SCORE_LIMIT

@cython_wrap
def test_gin_bonus(Environment test_env, test_agent):
    cdef Environment other_test_env = Environment(test_agent)
    assert other_test_env.GIN_BONUS == test_env.GIN_BONUS
    test_env.GIN_BONUS += 1
    assert other_test_env.GIN_BONUS == test_env.GIN_BONUS
    test_env.GIN_BONUS = 500
    assert other_test_env.GIN_BONUS == test_env.GIN_BONUS
    test_env.GIN_BONUS = 23
    assert other_test_env.GIN_BONUS == test_env.GIN_BONUS

@cython_wrap
def test_big_gin_bonus(Environment test_env, test_agent):
    cdef Environment other_test_env = Environment(test_agent)
    assert other_test_env.BIG_GIN_BONUS == test_env.BIG_GIN_BONUS
    test_env.BIG_GIN_BONUS += 1
    assert other_test_env.BIG_GIN_BONUS == test_env.BIG_GIN_BONUS
    test_env.BIG_GIN_BONUS = 500
    assert other_test_env.BIG_GIN_BONUS == test_env.BIG_GIN_BONUS
    test_env.BIG_GIN_BONUS = 23
    assert other_test_env.BIG_GIN_BONUS == test_env.BIG_GIN_BONUS

@cython_wrap
def test_player_1(Environment test_env):  #TODO test for when the new player isn't an actual player
    cdef Player test_player_1 = Player()
    cdef Player other_test_player_1 = Player()
    test_player_1.score = 0
    other_test_player_1.score = 0
    test_env.player_1 = test_player_1
    assert test_env.player_1 is test_player_1
    assert test_env.player_1 == test_player_1
    assert test_env.player_1 is not other_test_player_1
    test_player_1.score = 5
    assert test_env.player_1.score == 5
    assert test_env.player_1 is test_player_1
    test_env.player_1 = other_test_player_1
    assert test_env.player_1 is not test_player_1
    assert test_env.player_1.score == 0
    assert test_env.player_1 is other_test_player_1
