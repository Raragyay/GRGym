import pytest
from GRGym.environment.set cimport Set

from tests.utilities import cython_wrap

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
