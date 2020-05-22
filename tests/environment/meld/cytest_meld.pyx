import pytest
from GRGym.environment.meld cimport Meld

def cytest_meld_object():
    return Meld()

def cytest_connectable_cards(Meld meld):
    with pytest.raises(NotImplementedError):
        meld.connectable_cards()
