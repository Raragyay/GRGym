import numpy as np
import pytest

from GRGym.agent import BaseAgent


def test_act():
    test_agent = BaseAgent()
    with pytest.raises(NotImplementedError):
        test_agent.act(np.zeros(57))
