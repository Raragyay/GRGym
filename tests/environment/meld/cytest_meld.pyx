import pytest
from GRGym.environment.meld cimport Meld
from tests.utilities import cython_wrap

def base_meld():
    return Meld()

@pytest.fixture
def meld():
    return base_meld()

@cython_wrap
def test_connectable_cards(Meld meld):
    with pytest.raises(NotImplementedError):
        meld.connectable_cards()
