from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    card_state: np.ndarray
    player_score: int
    deck_size: int
    to_draw: bool
    to_discard: bool
