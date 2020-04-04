from pathlib import Path
from typing import List

import numpy as np
import pytest

from deadwood.deadwood_counter_dp import DeadwoodCounter


def retrieve_deadwood_val_tests(file_names=None):
    if file_names:
        test_data_file_names = Path(__file__).parent.glob("|".join(file_names))
    else:
        test_data_file_names = Path(__file__).parent.glob("td_*.txt")
    test_data = []
    for file_name in test_data_file_names:
        with file_name.open() as file:
            first_line = ""
            if "td_" in file_name.stem:
                test_name = file_name.stem[3:]
            else:
                test_name = file_name.stem

            def deck_generator():
                yield first_line
                for i in range(3):
                    yield file.readline()

            test_num = 1
            while True:
                first_line = file.readline()
                if not first_line.strip():
                    break
                # noinspection PyTypeChecker
                deck = np.nonzero(np.loadtxt(deck_generator(), dtype=np.bool).flatten())[0]
                expected = int(file.readline())
                test_id = f"{test_name}.{test_num}-{expected}"
                test_data.append(pytest.param(deck, expected, id=test_id))
                test_num += 1
    return test_data


def retrieve_deadwood_remaining_card_tests(file_names: List[str] = None):
    if file_names:
        test_data_file_names = Path(__file__).parent.glob("|".join(file_names))
    else:
        test_data_file_names = Path(__file__).parent.glob("tc_*.txt")
    test_data = []
    for file_name in test_data_file_names:
        with file_name.open() as file:
            first_line = ""
            test_name = file_name.stem[3:]

            def deck_generator():
                yield first_line
                for i in range(3):
                    yield file.readline()

            test_num = 1
            while True:
                first_line = file.readline()
                if not first_line.strip():
                    break
                # noinspection PyTypeChecker
                deck = np.nonzero(np.loadtxt(deck_generator(), dtype=np.bool).flatten())[0]
                expected = np.fromstring(file.readline(), dtype=np.int8, sep=" ")
                test_id = f"{test_name}.{test_num}"
                test_data.append(pytest.param(deck, expected, id=test_id))
                test_num += 1
    return test_data


@pytest.mark.parametrize("hand,expected_deadwood", retrieve_deadwood_val_tests())
def test_deadwood(hand: np.ndarray, expected_deadwood: int):
    counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood


@pytest.mark.parametrize("hand,expected_remaining_cards", retrieve_deadwood_remaining_card_tests())
def test_deadwood_remaining_cards(hand: np.ndarray, expected_remaining_cards: np.ndarray):
    counter = DeadwoodCounter(hand)
    assert set(counter.remaining_cards()) == set(expected_remaining_cards)


@pytest.mark.slow
@pytest.mark.parametrize("hand,expected_deadwood", retrieve_deadwood_val_tests(["slow_cases.txt"]))
def test_slowest(hand, expected_deadwood):
    counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood


@pytest.mark.parametrize("rank,expected_deadwood",
                         zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]))
def test_deadwood_val(rank, expected_deadwood):
    assert DeadwoodCounter.deadwood_val(rank) == expected_deadwood
