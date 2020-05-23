from .cytest_meld import *

def test_hash(meld):
    with pytest.raises(NotImplementedError):
        meld.__hash__()


def test_eq(meld):
    with pytest.raises(NotImplementedError):
        meld.__eq__(base_meld())
