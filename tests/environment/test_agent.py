import numpy as np
import pytest

from environment.agent import Agent


def test_act():
    test_agent = Agent()
    with pytest.raises(NotImplementedError):
        test_agent.act(np.zeros(57))
