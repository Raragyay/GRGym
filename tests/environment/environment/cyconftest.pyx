include "cython_wrapper.pxi"
from GRGym.environment.player cimport Player
cimport numpy as np

@cython_wrap
def player_with_cards(Player test_player):
    def player_factory(np.ndarray card_list):
        for card in card_list:
            test_player.add_card_from_deck(card)
        return test_player
    return player_factory
