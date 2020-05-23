include "cython_wrapper.pxi"
import pytest

from GRGym.environment.meld cimport Meld

def base_meld():
    return Meld()

@pytest.fixture
def meld():
    return base_meld()

@cython_wrap
def test_connectable_cards(Meld meld):
    with pytest.raises(NotImplementedError):
        meld.connectable_cards()

@cython_wrap
def test_hash(meld):
    with pytest.raises(NotImplementedError):
        meld.__hash__()

@cython_wrap
def test_eq(meld):
    with pytest.raises(NotImplementedError):
        meld.__eq__(base_meld())
