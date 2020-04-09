import numpy as np
import pytest
from agent.baseagent import BaseAgent

from environment.environment import Environment
from environment.player import Player
from tests.utilities import idfn_id_expected, retrieve_boolean, retrieve_file_tests, retrieve_nonzero_indices


@pytest.fixture
def test_env():
    return Environment(BaseAgent())  # TODO change to testing agent maybe?


@pytest.fixture
def base_player():
    test_player = Player()
    return test_player


@pytest.fixture()
def player_with_cards(base_player: Player):
    def player_factory(card_list: np.ndarray):
        for card in card_list:
            base_player.add_card_from_deck(card)
        return base_player

    return player_factory


@pytest.mark.parametrize("cards_in_hand,expected", retrieve_file_tests(retrieve_nonzero_indices, retrieve_boolean,
                                                                       idfn_id_expected,
                                                                       file_names=["environment/can_knock_cases.txt"]))
def test_can_knock(test_env: Environment, player_with_cards, cards_in_hand: list, expected: bool):
    assert test_env.can_knock(player_with_cards(cards_in_hand)) == expected


@pytest.mark.parametrize("cards_in_hand,expected", retrieve_file_tests(retrieve_nonzero_indices, retrieve_boolean,
                                                                       idfn_id_expected,
                                                                       file_names=["environment/is_gin.txt"]))
def test_is_gin(test_env: Environment, player_with_cards, cards_in_hand: list, expected: bool):
    assert test_env.is_gin(player_with_cards(cards_in_hand)) == expected


@pytest.mark.parametrize("actions,expected", [(np.zeros(56), True)])
def test_wants_to_knock(test_env: Environment, actions: np.ndarray, expected: bool):
    assert test_env.wants_to_knock(actions) == expected
