import numpy as np

from src.agent.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.random.rand(56)
