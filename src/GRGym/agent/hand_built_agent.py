import numpy as np
from scipy.signal import convolve2d

from src.GRGym.agent import BaseAgent
from src.GRGym.environment.card_state import CardState


class HandBuiltAgent(BaseAgent):
    def act(self, observation: np.ndarray) -> np.ndarray:
        my_cards = (observation[0:52] == CardState.MINE_FROM_DECK) | (observation[0:52] == CardState.MINE_FROM_DISCARD)
        my_cards = my_cards.reshape((4, 13)).astype(np.int)
        convolve_filter = np.zeros((7, 7))
        for i in range(7):
            convolve_filter[i][3] = 1.5
        for i in range(1, 6):
            convolve_filter[3][i] = 1
        convolve_filter[3][2] = 2
        convolve_filter[3][4] = 2
        convolved = convolve2d(my_cards, convolve_filter, mode='same').reshape(52, )
        # print(my_cards)
        # print(convolve_filter)
        # print(convolved)
        # print(convolved.shape)
        result = np.empty((56,))
        result[0:52] = (-convolved)
        top_of_discard = np.where(
            (observation[0:52] == CardState.DISCARD_MINE_TOP) | (observation[0:52] == CardState.DISCARD_THEIRS_TOP))
        # print(top_of_discard)
        result[52] = convolved[top_of_discard[0][0]] if len(top_of_discard[0]) else 0
        result[53] = np.mean(convolved[observation[0:52] == CardState.UNKNOWN])
        result[54] = 1
        result[55] = 0
        # print(result)
        return result
