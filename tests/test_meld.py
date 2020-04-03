import pytest

from deadwood.meld import Meld


def test_connectable_cards():
    meld = Meld()
    with pytest.raises(NotImplementedError):
        meld.connectable_cards()
