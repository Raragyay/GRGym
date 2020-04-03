from pathlib import Path

import numpy as np
import pytest

from hand import Hand


def generate_has_card_data():
    test_data_file_names = list(Path(__file__).parent.glob("test_has_card.*.npy"))
    test_data = []
    for idx, file_name in enumerate(test_data_file_names):
        with file_name.open(mode="rb") as file:
            test_hand = Hand()
            # noinspection PyTypeChecker
            mask: np.ndarray = np.load(file)
            test_hand.cards = mask
            test_id = f"deck-{idx + 1}"
            test_data.append(pytest.param(test_hand, mask, id=test_id))

    return test_data


def card_shorthand_idfn(val):
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
                                                     f"K{Hand.suit_symbols[3]}")), ids=card_shorthand_idfn)
def test_card_shorthand(card_val: int, expected: str):
    assert Hand.card_shorthand(card_val) == expected


@pytest.mark.parametrize("starting_hand,expected", generate_has_card_data())
def test_has_card(starting_hand: Hand, expected: np.ndarray):
    for i in range(52):
        assert starting_hand.has_card(i) == expected[i]


def test_add_card():
    test_hand = Hand()
    for i in range(52):
        assert not test_hand.has_card(i)
        test_hand.add_card(i)
        assert test_hand.has_card(i)
        test_hand.add_card(i)
        assert test_hand.has_card(i)
    for i in range(52):
        assert test_hand.has_card(i)
        new_hand = Hand()
        assert not new_hand.has_card(i)
        new_hand.add_card(i)
        assert new_hand.has_card(i)


def test_remove_card():
    test_hand = Hand()
    for i in range(52):
        test_hand.add_card(i)
    for i in range(52):
        assert test_hand.has_card(i)
        test_hand.remove_card(i)
        assert not test_hand.has_card(i)


def test_card_list():
    test_hand = Hand()
    for i in range(52):
        test_hand.add_card(i)
        test_list = np.arange(i + 1)
        np.testing.assert_array_equal(test_hand.card_list(), test_list)
