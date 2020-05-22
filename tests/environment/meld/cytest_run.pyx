from GRGym.environment.run cimport Run
import pytest

def cytest_connectable_cards(Run run, expected):
    assert run.connectable_cards() == expected
