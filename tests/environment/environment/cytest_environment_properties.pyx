include "cython_wrapper.pxi"
import numpy as np
cimport numpy as np
import pytest
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
