import numpy as np
cimport numpy as np
from GRGym.agent import BaseAgent
from GRGym.environment.cythonenvironment cimport CythonEnvironment
from GRGym.environment.player cimport Player

def cytest_env():
    return CythonEnvironment(BaseAgent())  # TODO change to testing agent maybe?

def cybase_player():
    test_player = Player()
    return test_player

def cyplayer_with_cards(Player base_player):
    def player_factory(card_list: np.ndarray):
        for card in card_list:
            base_player.add_card_from_deck(card)
        return base_player
    return player_factory

def cytest_can_knock(CythonEnvironment test_env, player_with_cards, cards_in_hand: np.ndarray, expected: bool):
    assert CythonEnvironment.can_knock(player_with_cards(cards_in_hand)) == expected
