import numpy as np


class DeadwoodCounter:
    def __init__(self, hand: np.ndarray):
        self.hand = hand

    def deadwood(self):
        raise NotImplementedError("Do not call a base class. ")
