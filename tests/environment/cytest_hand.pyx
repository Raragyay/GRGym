include "cython_wrapper.pxi"
import numpy as np
cimport numpy as np
import pytest

from GRGym.environment.hand cimport Hand

@pytest.fixture
def test_hand():
    test_hand = Hand()
    return test_hand

@cython_wrap
def idfn_card_shorthand(val):
    if isinstance(val, int):
        return val
    else:
        return ""

@pytest.mark.parametrize("card_val,expected",
                         zip((1, 5, 9, 23, 46, 51), ("2D", "6D", "10D", "JC", "8S", "KS")), ids=idfn_card_shorthand)
@cython_wrap
def test_card_shorthand(card_val, expected):
    assert Hand.card_shorthand(card_val) == expected

@cython_wrap
def test_has_card(Hand test_hand):
    for i in range(52):
        assert (not test_hand.cards[i]) and (not test_hand.has_card(i))
        test_hand.cards[i] = True
        assert test_hand.has_card(i)
    np.testing.assert_array_equal(test_hand.cards == True, np.ones(52))

@cython_wrap
def test_add_card(Hand test_hand):
    for i in range(52):
        assert not test_hand.has_card(i)
        test_hand.add_card(i)
        assert test_hand.has_card(i)
        test_hand.add_card(i)
        assert test_hand.has_card(i)
    for i in range(52):
        assert test_hand.has_card(i)

@cython_wrap
def test_remove_card(Hand test_hand):
    for i in range(52):
        test_hand.add_card(i)
    for i in range(52):
        assert test_hand.has_card(i)
        test_hand.remove_card(i)
        assert not test_hand.has_card(i)

@cython_wrap
def test_eq(Hand test_hand):
    for i in range(52):
        new_hand = Hand()
        new_hand.cards = test_hand.cards.copy()
        new_hand.add_card(i)
        assert not test_hand == new_hand
        test_hand.add_card(i)
        assert test_hand == new_hand

@cython_wrap
def test_card_list(Hand test_hand):
    for i in range(52):
        test_hand.add_card(i)
        test_list = np.arange(i + 1)
        np.testing.assert_array_equal(test_hand.card_list(), test_list)