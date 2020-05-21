import numpy as np

from src.GRGym.agent import BaseAgent


class RandomAgent(BaseAgent):
    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.random.rand(56)
