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

@cython_wrap
def test_opponents(Environment test_env):
    cdef Player player_1_1 = Player()
    cdef Player player_1_2 = Player()
    cdef Player player_2_1 = Player()
    cdef Player player_2_2 = Player()

    # Assert initial configuration is working
    assert test_env.opponents(test_env.player_1) is test_env.player_2
    assert test_env.opponents(test_env.player_2) is test_env.player_1

    # Assert changing player 1 maintains opponent referencing
    test_env.player_1 = player_1_1
    assert test_env.opponents(player_1_1) is test_env.player_2
    assert test_env.opponents(test_env.player_2) is player_1_1

    # Assert changing player 2 maintains opponent referencing
    test_env.player_2 = player_2_1
    assert test_env.opponents(player_1_1) is player_2_1
    assert test_env.opponents(player_2_1) is player_1_1

    # Ensure that getting the opponent of a player that isn't recorded in the environment raises an error
    with pytest.raises(ValueError):
        test_env.opponents(player_1_2)
    with pytest.raises(ValueError):
        test_env.opponents(player_2_2)

@cython_wrap
def test_discard_pile_is_empty(Environment test_env):
    for i in range(52):
        test_env.discard_pile = np.arange(i, dtype=np.int8)
        assert i == 0 or not test_env.discard_pile_is_empty()
        test_env.discard_pile = np.zeros(i, dtype=np.int8)
        assert i == 0 or not test_env.discard_pile_is_empty()

@cython_wrap
def test_repr(Environment test_env):
    assert repr(test_env)  # make sure no errors

@cython_wrap
def test_get_opponent_deadwood(Environment test_env):
    assert Environment.get_deadwood(test_env.player_1) == test_env.get_opponent_deadwood(test_env.player_2)
    assert test_env.get_opponent_deadwood(test_env.player_1) == Environment.get_deadwood(test_env.player_2)
