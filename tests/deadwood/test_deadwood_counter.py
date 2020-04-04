from pathlib import Path
from typing import Callable, List, TextIO

import numpy as np
import pytest

from deadwood.deadwood_counter_dp import DeadwoodCounter


def retrieve_deadwood_tests(expected_func: Callable, id_func: Callable,
                            file_suffix: str = None, file_names: List[str] = None):
    if file_names:
        test_data_file_names = Path(__file__).parent.glob("|".join(file_names))
    elif file_suffix:
        test_data_file_names = Path(__file__).parent.glob(f"{file_suffix}*.txt")
    else:
        raise ValueError("Please provide a file suffix or a list of files to parse.")
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
                expected = expected_func(file)
                test_id: str = id_func(test_name, test_num, expected)
                test_data.append(pytest.param(deck, expected, id=test_id))
                test_num += 1
    return test_data


def deadwood_expected(file: TextIO):
    return int(file.readline())


def deadwood_id(test_name, test_num, expected):
    return f"{test_name}.{test_num}-{expected}"


def deadwood_remaining_cards_expected(file):
    return list(map(int, file.readline().split()))


def deadwood_remaining_cards_id(test_name, test_num, expected):
    return f"{test_name}.{test_num}"


@pytest.mark.parametrize("hand,expected_deadwood", retrieve_deadwood_tests(deadwood_expected, deadwood_id,
                                                                           file_suffix="td_"))
def test_deadwood(hand: np.ndarray, expected_deadwood: int):
    counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood


@pytest.mark.parametrize("hand,expected_remaining_cards", retrieve_deadwood_tests(deadwood_remaining_cards_expected,
                                                                                  deadwood_remaining_cards_id,
                                                                                  file_suffix="tc_"))
def test_deadwood_remaining_cards(hand: np.ndarray, expected_remaining_cards: List[int]):
    counter = DeadwoodCounter(hand)
    assert set(counter.remaining_cards()) == set(expected_remaining_cards)


@pytest.mark.slow
@pytest.mark.parametrize("hand,expected_deadwood", retrieve_deadwood_tests(deadwood_expected,
                                                                           deadwood_id, file_names=["slow_cases.txt"]))
def test_slowest(hand, expected_deadwood):
    counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood


@pytest.mark.parametrize("rank,expected_deadwood",
                         zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]))
def test_deadwood_val(rank, expected_deadwood):
    assert DeadwoodCounter.deadwood_val(rank) == expected_deadwood
