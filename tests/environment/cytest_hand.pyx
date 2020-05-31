include "cython_wrapper.pxi"
import numpy as np
cimport numpy as np
import pytest

from GRGym.environment.hand cimport Hand

def base_hand():
    return Hand()

@pytest.fixture
def test_hand():
    return base_hand()

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
        test_hand.__cards[i] = True
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
def test_eq_cards(Hand test_hand):
    for i in range(52):
        new_hand = Hand()
        new_hand.cards = test_hand.cards
        new_hand.add_card(i)
        assert not test_hand == new_hand
        test_hand.add_card(i)
        assert test_hand == new_hand

@cython_wrap
def test_eq_type(Hand test_hand):
    cdef int wrong_type_int = 2
    cdef str wrong_type_str = "fake"
    assert test_hand != wrong_type_int
    assert test_hand != wrong_type_str

@cython_wrap
def test_str_repr(Hand test_hand):
    #This is just to test that the str and repr functions return something.
    blank_str = str(test_hand)
    blank_repr = repr(test_hand)
    test_hand.add_card(3)
    test_hand.add_card(10)
    assert str(test_hand) != blank_str  # We want the representation to mean something
    assert repr(test_hand) != blank_repr
    repr_with_data = repr(test_hand)
    test_hand.add_card(17)
    assert repr_with_data != repr(test_hand)  # Different hands should have different repr

@cython_wrap
def test_card_list(Hand test_hand):
    for i in range(52):
        test_hand.add_card(i)
        test_list = np.arange(i + 1)
        np.testing.assert_array_equal(test_hand.card_list(), test_list)

@cython_wrap
def test_copy(Hand test_hand):
    cdef Hand new_hand
    for i in range(51, -1, -1):
        test_hand.add_card(i)
        new_hand = test_hand.copy()
        assert test_hand == new_hand
    for i in range(0, 52):
        test_hand.remove_card(i)
        new_hand = test_hand.copy()
        assert test_hand == new_hand

@cython_wrap
def test_cards_property(Hand test_hand):
    cdef int i, j
    cdef np.ndarray mock_card_state = np.zeros(52, dtype=np.bool)
    for i in range(52):
        assert test_hand.__cards[i] == 0
        assert test_hand.cards[i] == 0
        test_hand.cards = mock_card_state
        assert test_hand.cards[i] == 0
        assert test_hand.__cards[i] == 0

        mock_card_state[i] = 1
        assert test_hand.cards[i] == 0  # hand should make a copy of the parameter given
        assert test_hand.__cards[i] == 0

        mock_card_state[i] = 0
        test_hand.cards[i] = 1
        assert test_hand.__cards[i] == 0  # hand __get__ property should be a copy and not a reflection of the
        # internal array
        assert mock_card_state[i] == 0

        test_hand.__cards[i] = 1
        assert test_hand.cards[i] == 1
        assert mock_card_state[i] == 0
        mock_card_state[i] = 1
