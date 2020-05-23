from .cytest_set import *

def test_eq(set_class):
    assert set_class(1) != 0
    assert set_class(1) != set_class(2)
    assert set_class(2) == set_class(2)
