from typing import Set

import pytest

from GRGym.environment.run import Run


def build_test_data():
    test_data = []
    start_ends = [(0, 2), (3, 5), (2, 6), (5, 10), (10, 12), (6, 9), (0, 12)]
    expecteds = [(3,), (2, 6), (1, 7), (4, 11), (9,), (5, 10), tuple()]
    for suit in range(4):
        for idx in range(len(start_ends)):
            start = start_ends[idx][0] + 13 * suit
            end = start_ends[idx][1] + 13 * suit
            expected = {suit * 13 + x for x in expecteds[idx]}
            test_data.append(
                pytest.param(start, end, expected,
                             id=f"{start},{end}-{expected}"))
    return test_data


@pytest.mark.parametrize("start,end,expected", build_test_data())
def test_connectable_cards(start: int, end: int, expected: Set[int]):
    test_run = Run(start, end)
    assert test_run.connectable_cards() == expected
