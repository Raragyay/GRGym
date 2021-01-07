import numpy as np
import pytest

from GRGym.agent import BaseAgent, RandomAgent


def test_act():
    test_agent = BaseAgent()
    with pytest.raises(NotImplementedError):
        test_agent.act(np.zeros(57))


def test_reset():
    test_agent = BaseAgent()
    with pytest.raises(NotImplementedError):
        test_agent.reset()


def test_random():
    test_agent = RandomAgent()
    np.random.seed(52)
    test_arr = np.random.rand(56)
    np.random.seed(52)
    given_arr = test_agent.act(np.zeros(57))
    np.testing.assert_array_equal(test_arr, given_arr)
    assert test_agent.reset() is None  # it shouldn't do anything
