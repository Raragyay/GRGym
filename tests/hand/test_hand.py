from pathlib import Path

import numpy as np
import pytest

from hand import Hand


def generate_has_card_data():
    test_data_file_names = list(Path(__file__).parent.glob("test_has_card.*.npy"))
    print("hi")
    print(test_data_file_names)
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

# def test_add_card():
#     assert False
#
#
# def test_remove_card():
#     assert False
#
#
# def test_card_list():
#     assert False
