from GRGym.environment.meld import Meld

from tests.environment.meld.cytest_meld import *


@pytest.fixture
def meld():
    return cytest_meld_object()


def test_meld(meld):
    cytest_connectable_cards(meld)


def test_hash(meld):
    with pytest.raises(NotImplementedError):
        meld.__hash__()


def test_eq(meld):
    with pytest.raises(NotImplementedError):
        meld.__eq__(cytest_meld_object())
