from copy import deepcopy

import numpy as np
import pytest

from GRGym.environment.card_state cimport CardState
from GRGym.environment.player cimport Player

def cytest_player():
    return Player()

def cytest_reset(Player test_player):
    np.testing.assert_array_equal(test_player.hand_mask(), np.zeros(52))
    np.testing.assert_array_equal(test_player.card_states, np.zeros(52))
    test_player.add_card_from_deck(23)
    test_player.add_card_from_deck(42)
    test_player.add_card_from_deck(1)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(test_player.hand_mask(), np.zeros(52))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(test_player.card_states, np.zeros(52))
    test_player.reset_hand()
    np.testing.assert_array_equal(test_player.hand_mask(), np.zeros(52))
    np.testing.assert_array_equal(test_player.card_states, np.zeros(52))

def cytest_add_card_from_deck(Player test_player):
    for i in range(52):
        assert test_player.card_states[i] == CardState.UNKNOWN
        test_player.add_card_from_deck(i)
        assert test_player.card_states[i] == CardState.MINE_FROM_DECK
        assert i in test_player.card_list()
        test_player.add_card_from_deck(i)
        assert test_player.card_states[i] == CardState.MINE_FROM_DECK

def cytest_has_card(Player test_player):
    for i in range(52):
        assert not test_player.has_card(i)
        test_player.add_card_from_deck(i)
        assert test_player.has_card(i)

def cytest_card_list(Player test_player):
    cards_to_add = [2, 6, 23, 51, 11, 0, 45, 26, 32]
    for i in range(len(cards_to_add)):
        test_player.add_card_from_deck(cards_to_add[i])
        np.testing.assert_array_equal(test_player.card_list(), np.fromiter(sorted(cards_to_add[:i + 1]), np.int8))

def cytest_hand_mask(Player test_player):
    cards_to_add = [2, 6, 23, 51, 11, 0, 45, 26, 32]
    model_mask = np.zeros(52, np.bool)
    for i in range(len(cards_to_add)):
        test_player.add_card_from_deck(cards_to_add[i])
        model_mask[cards_to_add[i]] = True
        np.testing.assert_array_equal(test_player.hand_mask(), model_mask)

def cytest_update_card_to_top(Player test_player):
    for i in range(52):
        test_player.card_states[i] = CardState.DISCARD_MINE
        test_player.update_card_to_top(i)
        assert test_player.card_states[i] == CardState.DISCARD_MINE_TOP

        test_player.card_states[i] = CardState.DISCARD_THEIRS
        test_player.update_card_to_top(i)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS_TOP

def cytest_update_card_down(Player test_player):
    for i in range(52):
        test_player.card_states[i] = CardState.DISCARD_MINE_TOP
        test_player.update_card_down(i)
        assert test_player.card_states[i] == CardState.DISCARD_MINE

        test_player.card_states[i] = CardState.DISCARD_THEIRS_TOP
        test_player.update_card_down(i)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS

def cytest_discard_card(Player test_player):
    test_player.card_states[0] = CardState.DISCARD_MINE_TOP
    for i in range(1, 52):  # Use 0 as the initial discard card
        test_player.add_card_from_deck(i)
    for i in range(1, 52):
        test_player.discard_card(i, i - 1)
        assert test_player.card_states[i] == CardState.DISCARD_MINE_TOP
        assert test_player.card_states[i - 1] == CardState.DISCARD_MINE
        assert not test_player.has_card(i)

def cytest_report_opponent_discarded(Player test_player):
    test_player.card_states[0] = CardState.DISCARD_THEIRS_TOP
    for i in range(1, 52):
        test_player.report_opponent_discarded(i, i - 1)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS_TOP
        assert test_player.card_states[i - 1] == CardState.DISCARD_THEIRS

def cytest_add_card_from_discard(Player test_player):
    test_player.card_states.fill(CardState.DISCARD_MINE)
    test_player.card_states[51] = CardState.DISCARD_MINE_TOP
    for i in range(51, 1, -1):
        test_player.add_card_from_discard(i, i - 1)
        assert test_player.card_states[i] == CardState.MINE_FROM_DISCARD
        assert test_player.card_states[i - 1] == CardState.DISCARD_MINE_TOP
        assert test_player.card_states[i - 2] == CardState.DISCARD_MINE
        assert test_player.has_card(i)

def cytest_report_opponent_drew_from_discard(Player test_player):
    test_player.card_states.fill(CardState.DISCARD_MINE)
    test_player.card_states[51] = CardState.DISCARD_MINE_TOP
    for i in range(51, 1, -1):
        test_player.report_opponent_drew_from_discard(i, i - 1)
        assert test_player.card_states[i] == CardState.THEIRS_FROM_DISCARD
        assert test_player.card_states[i - 1] == CardState.DISCARD_MINE_TOP
        assert test_player.card_states[i - 2] == CardState.DISCARD_MINE
        assert not test_player.has_card(i)

def cytest_copy(Player test_player):
    cdef Player copy_player = test_player.copy()
    assert copy_player.hand == test_player.hand
    assert copy_player.score == test_player.score
    np.testing.assert_array_equal(copy_player.card_states, test_player.card_states)

def cytest_eq(Player test_player):
    cdef Player temp_player
    for i in range(52):
        temp_player = test_player.copy()
        temp_player.add_card_from_deck(i)
        assert not test_player == temp_player
        test_player.add_card_from_deck(i)
        assert test_player == temp_player
    for i in range(52):
        temp_player = test_player.copy()
        temp_player.add_card_from_discard(i, 0)
        assert not test_player == temp_player
        test_player.add_card_from_discard(i, 0)
        assert test_player == temp_player
    temp_player = test_player.copy()
    temp_player.score = 5
    assert not test_player == temp_player
    test_player.score = 5
    assert test_player == temp_player
