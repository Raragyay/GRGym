from pathlib import Path

import numpy as np
import pytest

from deadwood.deadwood_counter_dp import DeadwoodCounter


def retrieve_deadwood_tests(file_names=None):
    test_data_file_names = Path(__file__).parent.glob("|".join(file_names)) if file_names else Path(
        __file__).parent.glob("td_*.txt")
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


@pytest.mark.parametrize("hand,expected_deadwood", retrieve_deadwood_tests())
def test_deadwood(hand, expected_deadwood):
    counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood


@pytest.mark.slow
@pytest.mark.parametrize("hand,expected_deadwood", retrieve_deadwood_tests(["slow_cases.txt"]))
def test_slowest(hand, expected_deadwood):
    counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood


@pytest.mark.parametrize("rank,expected_deadwood",
                         zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]))
def test_deadwood_val(rank, expected_deadwood):
    assert DeadwoodCounter.deadwood_val(rank) == expected_deadwood
