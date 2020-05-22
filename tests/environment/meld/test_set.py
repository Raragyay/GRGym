from .cytest_set import *


@pytest.fixture
def set_class():
    return cytest_set_class()


def test_connectable_cards():
    cytest_connectable_cards()


def test_eq(set_class):
    assert set_class(1) != 0
    assert set_class(1) != set_class(2)
    assert set_class(2) == set_class(2)
