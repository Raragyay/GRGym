import numpy as np
import pytest

from environment.hand import Hand


@pytest.fixture
def test_hand():
    test_hand = Hand()
    return test_hand


def idfn_card_shorthand(val):
    if isinstance(val, int):
        return val
    else:
        return ""


@pytest.mark.parametrize("card_val,expected",
                         zip((1, 5, 9, 23, 46, 51), (f"2{Hand.suit_symbols[0]}",
                                                     f"6{Hand.suit_symbols[0]}",
                                                     f"10{Hand.suit_symbols[0]}",
                                                     f"J{Hand.suit_symbols[1]}",
                                                     f"8{Hand.suit_symbols[3]}",
                                                     f"K{Hand.suit_symbols[3]}")), ids=idfn_card_shorthand)
def test_card_shorthand(card_val: int, expected: str):
    assert Hand.card_shorthand(card_val) == expected


def test_has_card(test_hand: Hand):
    for i in range(52):
        assert (not test_hand.cards[i]) and (not test_hand.has_card(i))
        test_hand.cards[i] = True
        assert test_hand.has_card(i)
    np.testing.assert_array_equal(test_hand.cards == True, np.ones(52))


def test_add_card(test_hand: Hand):
    for i in range(52):
        assert not test_hand.has_card(i)
        test_hand.add_card(i)
        assert test_hand.has_card(i)
        test_hand.add_card(i)
        assert test_hand.has_card(i)
    for i in range(52):
        assert test_hand.has_card(i)


def test_remove_card(test_hand: Hand):
    for i in range(52):
        test_hand.add_card(i)
    for i in range(52):
        assert test_hand.has_card(i)
        test_hand.remove_card(i)
        assert not test_hand.has_card(i)


def test_card_list(test_hand: Hand):
    for i in range(52):
        test_hand.add_card(i)
        test_list = np.arange(i + 1)
        np.testing.assert_array_equal(test_hand.card_list(), test_list)


def test_eq(test_hand: Hand):
    for i in range(52):
        new_hand = Hand()
        new_hand.cards = test_hand.cards.copy()
        new_hand.add_card(i)
        assert not test_hand == new_hand
        test_hand.add_card(i)
        assert test_hand == new_hand
