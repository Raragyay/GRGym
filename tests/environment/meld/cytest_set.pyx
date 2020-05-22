import pytest
from GRGym.environment.set cimport Set

def cytest_set_class():
    return Set

def cytest_connectable_cards():
    for i in range(13):
        test_set = Set(i)
        assert test_set.connectable_cards() == {suit * 13 + i for suit in range(4)}
