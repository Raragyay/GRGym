include "cython_wrapper.pxi"
import numpy as np
cimport numpy as np
import pytest
from GRGym.environment.environment cimport Environment
from GRGym.environment.player cimport Player
from GRGym.environment.action_result cimport ActionResult
from tests.utilities import idfn_id_expected, retrieve_boolean, retrieve_file_tests, \
    retrieve_float_vector, retrieve_int, retrieve_nonzero_indices

"""
When testing static methods, using fixtures to pass the class name is ineffective because the cdef functions are not 
discoverable from Python code. Cython static methods can only be called from the class name and not from an object. 
"""
def generate_player_with_cards_func(Player player):
    def player_factory(np.ndarray card_list):
        for card in card_list:
            player.add_card_from_deck(card)
        return player
    return player_factory

@pytest.mark.parametrize("cards_in_hand,expected", retrieve_file_tests(retrieve_nonzero_indices, retrieve_boolean,
                                                                       idfn_id_expected,
                                                                       file_names=[
                                                                           "environment/environment/can_knock_cases.txt"]))
@cython_wrap
def test_can_knock(player_with_cards, np.ndarray cards_in_hand, bint expected):
    assert Environment.can_knock(
        player_with_cards(cards_in_hand)) == expected, f"{player_with_cards(cards_in_hand)} | {expected}"

@pytest.mark.parametrize("cards_in_hand,expected", retrieve_file_tests(retrieve_nonzero_indices, retrieve_boolean,
                                                                       idfn_id_expected,
                                                                       file_names=[
                                                                           "environment/environment/is_gin_cases.txt"]))
@cython_wrap
def test_is_gin(player_with_cards, np.ndarray cards_in_hand, bint expected):
    assert Environment.is_gin(player_with_cards(cards_in_hand)) == expected

@pytest.mark.parametrize("actions,expected", retrieve_file_tests(retrieve_float_vector, retrieve_boolean,
                                                                 idfn_id_expected,
                                                                 file_names=[
                                                                     "environment/environment/wants_to_knock_cases.txt"]))
@cython_wrap
def test_wants_to_knock(np.ndarray actions, bint expected):
    assert Environment.wants_to_knock(actions) == expected

@pytest.mark.parametrize("actions,expected", retrieve_file_tests(retrieve_float_vector, retrieve_boolean,
                                                                 idfn_id_expected,
                                                                 file_names=[
                                                                     "environment/environment/wants_to_draw_from_deck_cases.txt"]))
@cython_wrap
def test_wants_to_draw_from_deck(np.ndarray actions, bint expected):
    assert Environment.wants_to_draw_from_deck(actions) == expected

@cython_wrap
def test_update_score(Environment test_env, Player test_player):
    score_limit = test_env.SCORE_LIMIT
    assert test_env.update_score(test_player, score_limit // 2) == ActionResult.WON_HAND
    assert test_env.update_score(test_player, score_limit) == ActionResult.WON_MATCH
    assert test_player.score >= score_limit
    test_player.score = 0
    assert test_env.update_score(test_player, score_limit == ActionResult.WON_MATCH)

@pytest.mark.parametrize("cards_in_hand,deadwood", retrieve_file_tests(retrieve_nonzero_indices, retrieve_int,
                                                                       idfn_id_expected,
                                                                       file_names=["environment/deadwood/td_10.txt"]))
@cython_wrap
def test_score_gin(Environment test_env, Player test_player, player_with_cards, np.ndarray cards_in_hand,
                   int deadwood):
    test_env.player_1 = test_player
    test_env.player_2 = player_with_cards(cards_in_hand)
    test_player.score = test_env.SCORE_LIMIT - test_env.GIN_BONUS - deadwood
    assert test_env.score_gin(test_player) == ActionResult.WON_MATCH
    test_player.score = test_env.SCORE_LIMIT - test_env.GIN_BONUS - deadwood - 1
    assert test_env.score_gin(test_player) == ActionResult.WON_HAND

@pytest.mark.parametrize("cards_in_hand,deadwood", retrieve_file_tests(retrieve_nonzero_indices, retrieve_int,
                                                                       idfn_id_expected,
                                                                       file_names=["environment/deadwood/td_10.txt"]))
@cython_wrap
def test_score_big_gin(Environment test_env, Player test_player, player_with_cards, np.ndarray cards_in_hand,
                       int deadwood):
    test_env.player_1 = test_player
    test_env.player_2 = player_with_cards(cards_in_hand)
    test_player.score = test_env.SCORE_LIMIT - test_env.BIG_GIN_BONUS - deadwood
    assert test_env.score_big_gin(test_player) == ActionResult.WON_MATCH
    test_player.score = test_env.SCORE_LIMIT - test_env.BIG_GIN_BONUS - deadwood - 1
    assert test_env.score_big_gin(test_player) == ActionResult.WON_HAND

# TODO test discard pile resizing
