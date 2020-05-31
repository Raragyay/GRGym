include "cython_wrapper.pxi"
from GRGym.environment.run cimport Run
import pytest

def base_run_class():
    return Run

@pytest.fixture
def run_class():
    return base_run_class()

def build_test_data():
    test_data = []
    start_ends = [(0, 2), (3, 5), (2, 6), (5, 10), (10, 12), (6, 9), (0, 12)]
    expected_connectables = [(3,), (2, 6), (1, 7), (4, 11), (9,), (5, 10), tuple()]
    for suit in range(4):
        for idx in range(len(start_ends)):
            start = start_ends[idx][0] + 13 * suit
            end = start_ends[idx][1] + 13 * suit
            expected = {suit * 13 + x for x in expected_connectables[idx]}
            test_data.append(
                pytest.param(start, end, expected,
                             id=f"{start},{end}-{expected}"))
    return test_data

@pytest.mark.parametrize("start, end, expected", build_test_data())
@cython_wrap
def test_connectable_cards(run_class, start, end, expected):
    cdef Run run = run_class(start, end)
    assert run.connectable_cards() == expected

@cython_wrap
def test_eq(run_class):
    assert run_class(1, 3) != 0
    assert run_class(1, 3) != "string"
    assert run_class(1, 2) != run_class(2, 3)
    assert run_class(2, 4) != run_class(3, 4)
    assert run_class(2, 5) == run_class(2, 5)

@cython_wrap
def test_str_repr(run_class):
    cdef Run run = run_class(0, 2)
    assert str(run)  #Test that no errors occur
    assert repr(run)
    zero_two_string = repr(run)
    run = run_class(31, 35)
    assert repr(run) != zero_two_string
