import pytest
from GRGym.environment.set import Set


def test_connectable_cards():
    for i in range(13):
        test_set = Set(i)
        assert test_set.connectable_cards() == {suit * 13 + i for suit in range(4)}


def test_eq():
    with pytest.raises(AssertionError):
        assert Set(1) == 0
    assert Set(2) == Set(2)
