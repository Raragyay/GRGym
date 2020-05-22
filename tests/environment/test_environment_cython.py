from typing import Callable

import numpy as np
import pytest

from GRGym.agent import BaseAgent
from GRGym.environment.action_result import ActionResult
from GRGym.environment.cythonenvironment import CythonEnvironment
from GRGym.environment.player import Player
from tests.utilities import idfn_id_expected, retrieve_boolean, retrieve_file_tests, retrieve_float_vector, \
    retrieve_int, retrieve_nonzero_indices
from .cytest_environment_cython import *


@pytest.fixture
def test_env():
    return cytest_env()


@pytest.fixture
def base_player():
    return cybase_player()


@pytest.fixture()
def player_with_cards(base_player: Player):
    return cyplayer_with_cards(base_player)


@pytest.mark.parametrize("cards_in_hand,expected", retrieve_file_tests(retrieve_nonzero_indices, retrieve_boolean,
                                                                       idfn_id_expected,
                                                                       file_names=["environment/can_knock_cases.txt"]))
def test_can_knock(test_env: CythonEnvironment, player_with_cards: Callable[[np.ndarray], Player],
                   cards_in_hand: np.ndarray,
                   expected: bool):
    cytest_can_knock(test_env, player_with_cards, cards_in_hand, expected)


@pytest.mark.parametrize("cards_in_hand,expected", retrieve_file_tests(retrieve_nonzero_indices, retrieve_boolean,
                                                                       idfn_id_expected,
                                                                       file_names=["environment/is_gin_cases.txt"]))
def test_is_gin(test_env: CythonEnvironment, player_with_cards: Callable[[np.ndarray], Player],
                cards_in_hand: np.ndarray,
                expected: bool):
    assert CythonEnvironment.is_gin(player_with_cards(cards_in_hand)) == expected


@pytest.mark.parametrize("actions,expected", retrieve_file_tests(retrieve_float_vector, retrieve_boolean,
                                                                 idfn_id_expected,
                                                                 file_names=["environment/wants_to_knock_cases.txt"]))
def test_wants_to_knock(test_env: CythonEnvironment, actions: np.ndarray, expected: bool):
    assert test_env.wants_to_knock(actions) == expected


def test_update_score(test_env: CythonEnvironment, base_player: Player):
    score_limit = test_env.SCORE_LIMIT
    assert test_env.update_score(base_player, score_limit // 2) == ActionResult.WON_HAND
    assert test_env.update_score(base_player, score_limit) == ActionResult.WON_MATCH
    assert base_player.score >= score_limit
    base_player.score = 0
    assert test_env.update_score(base_player, score_limit == ActionResult.WON_MATCH)


@pytest.mark.parametrize("cards_in_hand,deadwood", retrieve_file_tests(retrieve_nonzero_indices, retrieve_int,
                                                                       idfn_id_expected,
                                                                       file_names=["environment/deadwood/td_10.txt"]))
def test_score_gin(test_env: CythonEnvironment, base_player: Player, player_with_cards: Callable[[np.ndarray], Player],
                   cards_in_hand: np.ndarray, deadwood: int):
    test_env.player_1 = base_player
    test_env.player_2 = player_with_cards(cards_in_hand)
    base_player.score = test_env.SCORE_LIMIT - test_env.GIN_BONUS - deadwood
    assert test_env.score_gin(base_player) == ActionResult.WON_MATCH
    base_player.score = test_env.SCORE_LIMIT - test_env.GIN_BONUS - deadwood - 1
    assert test_env.score_gin(base_player) == ActionResult.WON_HAND
