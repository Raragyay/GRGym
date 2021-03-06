include "cython_wrapper.pxi"
import pytest
from GRGym.environment.deadwood_counter cimport DeadwoodCounter
from GRGym.environment.set cimport Set
from GRGym.environment.run cimport Run
cimport numpy as np

from tests.utilities import idfn_name_id, idfn_name_id_expected, retrieve_file_tests, retrieve_int, \
    retrieve_int_vector, retrieve_nonzero_indices

def melds_expected(str string):
    cdef set meld_set = set()
    cdef list meld_raw_arr = string.split()
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
@cython_wrap
def test_deadwood(np.ndarray hand, int expected_deadwood):
    cdef DeadwoodCounter counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood

@pytest.mark.parametrize("hand,expected_remaining_cards",
                         retrieve_file_tests(retrieve_nonzero_indices, retrieve_int_vector, idfn_name_id,
                                             file_suffix="environment/deadwood/tc_"))
@cython_wrap
def test_remaining_cards(np.ndarray hand, np.ndarray expected_remaining_cards):
    cdef DeadwoodCounter counter = DeadwoodCounter(hand)
    assert counter.remaining_cards() == set(expected_remaining_cards)

@pytest.mark.slow
@pytest.mark.parametrize("hand,expected_deadwood",
                         retrieve_file_tests(retrieve_nonzero_indices, retrieve_int, idfn_name_id_expected,
                                             file_names=["environment/deadwood/slow_cases_td.txt"]))
@cython_wrap
def test_deadwood_slow(np.ndarray hand, int expected_deadwood):
    cdef DeadwoodCounter counter = DeadwoodCounter(hand)
    assert counter.deadwood() == expected_deadwood

@pytest.mark.parametrize("hand,expected_melds",
                         retrieve_file_tests(retrieve_nonzero_indices, melds_expected, idfn_name_id,
                                             file_suffix="environment/deadwood/tm_"))
@cython_wrap
def test_melds(np.ndarray hand, set expected_melds):
    cdef DeadwoodCounter counter = DeadwoodCounter(hand)
    assert counter.melds() == expected_melds, f'{counter.melds(), expected_melds}'

@pytest.mark.slow
@pytest.mark.parametrize("hand,expected_melds",
                         retrieve_file_tests(retrieve_nonzero_indices, melds_expected, idfn_name_id,
                                             file_names=["environment/deadwood/slow_cases_tm.txt"]))
@cython_wrap
def test_melds_slow(np.ndarray hand, set expected_melds):
    cdef DeadwoodCounter counter = DeadwoodCounter(hand)
    assert counter.melds() == expected_melds

@pytest.mark.parametrize("rank,expected_deadwood",
                         zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]))
@cython_wrap
def test_deadwood_val(rank, expected_deadwood):
    assert DeadwoodCounter.deadwood_val(rank) == expected_deadwood
