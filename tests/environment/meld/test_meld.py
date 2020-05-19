import pytest

from src.environment.meld import Meld


def test_connectable_cards():
    meld = Meld()
    with pytest.raises(NotImplementedError):
        meld.connectable_cards()


def test_hash():
    meld = Meld()
    with pytest.raises(NotImplementedError):
        meld.__hash__()


def test_eq():
    meld = Meld()
    with pytest.raises(NotImplementedError):
        meld.__eq__(Meld())
