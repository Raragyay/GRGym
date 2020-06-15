import numpy as np


class BaseAgent:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError()

    def act(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
