import typing

import numpy as np
import pytest
from GRGym.environment.meld import Meld
from GRGym.environment.run import Run
from GRGym.environment.set import Set
from tests.utilities import idfn_name_id, idfn_name_id_expected, retrieve_file_tests, retrieve_int, \
    retrieve_int_vector, \
    retrieve_nonzero_indices
from .cytest_deadwood_counter import *


@pytest.fixture
def deadwood_counter_class():
    return cytest_deadwood_counter_class()


def melds_expected(string: str):
    meld_set: typing.Set[Meld] = set()
    meld_raw_arr = string.split()
    for raw_meld in meld_raw_arr:
        if raw_meld[0] == "S":
            rank = int(raw_meld.split("-")[1])
            meld_set.add(Set(rank))
        elif raw_meld[0] == "R":
            raw_meld_split = raw_meld.split("-")
            suit = int(raw_meld_split[1])
            lower = int(raw_meld_split[2]) + 13 * suit
            upper = int(raw_meld_split[3]) + 13 * suit
            meld_set.add(Run(lower, upper))
        else:
            raise ValueError(f"Please provide expected melds in the correct format. Provided: {raw_meld}. Full "
                             f"string: {string}")
    return meld_set


@pytest.mark.parametrize("hand,expected_deadwood",
                         retrieve_file_tests(retrieve_nonzero_indices, retrieve_int, idfn_name_id_expected,
                                             file_suffix="environment/deadwood/td_"))
def test_deadwood(deadwood_counter_class, hand: np.ndarray, expected_deadwood: int):
    counter = deadwood_counter_class(hand)
    cytest_deadwood(counter, expected_deadwood)


@pytest.mark.parametrize("hand,expected_remaining_cards",
                         retrieve_file_tests(retrieve_nonzero_indices, retrieve_int_vector, idfn_name_id,
                                             file_suffix="environment/deadwood/tc_"))
def test_remaining_cards(deadwood_counter_class, hand: np.ndarray, expected_remaining_cards: np.ndarray):
    counter = deadwood_counter_class(hand)
    cytest_remaining_cards(counter, expected_remaining_cards)


@pytest.mark.parametrize("hand,expected_melds",
                         retrieve_file_tests(retrieve_nonzero_indices, melds_expected, idfn_name_id,
                                             file_suffix="environment/deadwood/tm_"))
def test_melds(deadwood_counter_class, hand: np.ndarray, expected_melds: typing.Set[Meld]):
    counter = deadwood_counter_class(hand)
    cytest_melds(counter, expected_melds)


@pytest.mark.slow
@pytest.mark.parametrize("hand,expected_deadwood",
                         retrieve_file_tests(retrieve_nonzero_indices, retrieve_int, idfn_name_id_expected,
                                             file_names=["environment/deadwood/slow_cases_td.txt"]))
def test_deadwood_slow(deadwood_counter_class, hand: np.ndarray, expected_deadwood: int):
    counter = deadwood_counter_class(hand)
    cytest_deadwood(counter, expected_deadwood)


@pytest.mark.slow
@pytest.mark.parametrize("hand,expected_melds",
                         retrieve_file_tests(retrieve_nonzero_indices, melds_expected, idfn_name_id,
                                             file_names=["environment/deadwood/slow_cases_tm.txt"]))
def test_melds_slow(deadwood_counter_class, hand: np.ndarray, expected_melds: typing.Set[Meld]):
    counter = deadwood_counter_class(hand)
    cytest_melds(counter, expected_melds)


@pytest.mark.parametrize("rank,expected_deadwood",
                         zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]))
def test_deadwood_val(deadwood_counter_class, rank: int, expected_deadwood: int):
    cytest_deadwood_val(deadwood_counter_class, rank, expected_deadwood)
