import typing

import numpy as np

from meld.meld import Meld


class DeadwoodCounter:
    def __init__(self, hand: np.ndarray):
        """
        Hand must be in suit then rank form, ascending order

        :param hand: A numpy array of the cards in the hand sorted in ascending order.  e.g. [2 25 36 47]
        """
        self.hand = hand

    def deadwood(self) -> int:
        raise NotImplementedError('Do not call the base class. ')

    def remaining_cards(self) -> typing.Set[int]:
        raise NotImplementedError('Do not call the base class. ')

    def melds(self) -> typing.Tuple[Meld, ...]:
        raise NotImplementedError('Do not call the base class. ')
