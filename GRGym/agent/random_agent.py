import numpy as np

from GRGym.agent import BaseAgent


class RandomAgent(BaseAgent):
    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.random.rand(56)

    def reset(self):
        pass
