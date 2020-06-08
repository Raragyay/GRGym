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

@cython_wrap
def test_player_2(Environment test_env):
    cdef Player test_player_2 = Player()
    cdef Player other_test_player_2 = Player()
    test_player_2.score = 0
    other_test_player_2.score = 0
    test_env.player_2 = test_player_2
    assert test_env.player_2 is test_player_2
    assert test_env.player_2 == test_player_2
    assert test_env.player_2 is not other_test_player_2
    test_player_2.score = 5
    assert test_env.player_2.score == 5
    assert test_env.player_2 is test_player_2
    test_env.player_2 = other_test_player_2
    assert test_env.player_2 is not test_player_2
    assert test_env.player_2.score == 0
    assert test_env.player_2 is other_test_player_2

@cython_wrap
def test_deck(Environment test_env):
    cdef np.ndarray test_deck = np.zeros((52,), dtype=np.int8)
    cdef np.ndarray other_test_deck = np.zeros((52,), dtype=np.int8)
    test_env.deck = test_deck
    for i in range(52):
        assert test_deck[i] == 0
        assert test_env.deck[i] == 0
        test_deck[i] = i
        assert test_env.deck[i] == i
    test_env.deck = other_test_deck
    np.testing.assert_array_equal(test_env.deck, other_test_deck)
    np.testing.assert_array_equal(test_env.__deck, other_test_deck)

    assert_environment_validates_property(test_env, "deck")

@cython_wrap
def test_discard_pile(Environment test_env):
    cdef np.ndarray test_discard = np.arange(52, dtype=np.int8)
    cdef np.ndarray other_test_discard = np.zeros(8, dtype=np.int8)
    test_env.discard_pile = test_discard
    assert len(test_env.discard_pile) == len(test_discard)
    assert len(test_env.__discard_pile) == len(test_discard)
    np.testing.assert_array_equal(test_env.discard_pile, test_discard)
    test_env.discard_pile = other_test_discard
    assert len(test_env.discard_pile) == len(other_test_discard)
    assert len(test_env.__discard_pile) == 52
    np.testing.assert_array_equal(test_env.discard_pile, other_test_discard)
    other_test_discard[0] = 5
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(test_env.discard_pile, other_test_discard)

    assert_environment_validates_property(test_env, "discard_pile")

def assert_environment_validates_property(Environment test_env, str object_property):
    cdef np.ndarray wrong_dim_array = np.arange(4, dtype=np.int8).reshape(2, 2)
    with pytest.raises(ValueError):
        test_env.__setattr__(object_property, wrong_dim_array)
    cdef np.ndarray wrong_size_array = np.arange(53, dtype=np.int8)
    with pytest.raises(ValueError):
        test_env.__setattr__(object_property, wrong_size_array)
    cdef np.ndarray wrong_value_array = np.arange(52, 53)
    with pytest.raises(ValueError):
        test_env.__setattr__(object_property, wrong_value_array)

@cython_wrap
def test_validate_card_array():
    # Test Error checks
    cdef np.ndarray wrong_dim_array = np.arange(4, dtype=np.int8).reshape(2, 2)
    with pytest.raises(ValueError):
        Environment.validate_card_array(wrong_dim_array)
    cdef np.ndarray wrong_size_array = np.arange(53, dtype=np.int8)
    with pytest.raises(ValueError):
        Environment.validate_card_array(wrong_size_array)
    cdef np.ndarray wrong_value_array = np.arange(52, 53, dtype=np.int8)
    with pytest.raises(ValueError):
        Environment.validate_card_array(wrong_value_array)

    cdef np.ndarray valid_array = np.arange(52, dtype=np.int8)
    Environment.validate_card_array(valid_array)
