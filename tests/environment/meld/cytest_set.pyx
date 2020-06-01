include "cython_wrapper.pxi"
import pytest
from GRGym.environment.set cimport Set

def base_set_class():
    return Set

@pytest.fixture
def set_class():
    return base_set_class()

@cython_wrap
def test_connectable_cards():
    for i in range(13):
        test_set = Set(i)
        assert test_set.connectable_cards() == {suit * 13 + i for suit in range(4)}

@cython_wrap
def test_eq(set_class):
    assert set_class(1) != 0
    assert set_class(1) != set_class(2)
    assert set_class(2) == set_class(2)

@cython_wrap
def test_str_repr(set_class):
    cdef Set set = set_class(0)
    assert str(set)  #Test that no errors occur
    assert repr(set)
    zero_string = repr(set)
    set = set_class(12)
    assert repr(set) != zero_string
