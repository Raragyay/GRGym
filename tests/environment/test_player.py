from copy import deepcopy

import numpy as np
import pytest

from GRGym.environment.card_state import CardState
from GRGym.environment.player import Player


@pytest.fixture
def test_player():
    test_player = Player()
    return test_player


def test_reset(test_player: Player):
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


def test_add_card_from_deck(test_player: Player):
    for i in range(52):
        assert test_player.card_states[i] == CardState.UNKNOWN
        test_player.add_card_from_deck(i)
        assert test_player.card_states[i] == CardState.MINE_FROM_DECK
        assert i in test_player.card_list()
        test_player.add_card_from_deck(i)
        assert test_player.card_states[i] == CardState.MINE_FROM_DECK


def test_has_card(test_player: Player):
    for i in range(52):
        assert not test_player.has_card(i)
        test_player.add_card_from_deck(i)
        assert test_player.has_card(i)


def test_card_list(test_player: Player):
    cards_to_add = [2, 6, 23, 51, 11, 0, 45, 26, 32]
    for i in range(len(cards_to_add)):
        test_player.add_card_from_deck(cards_to_add[i])
        np.testing.assert_array_equal(test_player.card_list(), np.fromiter(sorted(cards_to_add[:i + 1]), np.int8))


def test_hand_mask(test_player: Player):
    cards_to_add = [2, 6, 23, 51, 11, 0, 45, 26, 32]
    model_mask = np.zeros(52, np.bool)
    for i in range(len(cards_to_add)):
        test_player.add_card_from_deck(cards_to_add[i])
        model_mask[cards_to_add[i]] = True
        np.testing.assert_array_equal(test_player.hand_mask(), model_mask)


def test_update_card_to_top(test_player: Player):
    for i in range(52):
        test_player.card_states[i] = CardState.DISCARD_MINE
        test_player.update_card_to_top(i)
        assert test_player.card_states[i] == CardState.DISCARD_MINE_TOP

        test_player.card_states[i] = CardState.DISCARD_THEIRS
        test_player.update_card_to_top(i)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS_TOP


def test_update_card_down(test_player: Player):
    for i in range(52):
        test_player.card_states[i] = CardState.DISCARD_MINE_TOP
        test_player.update_card_down(i)
        assert test_player.card_states[i] == CardState.DISCARD_MINE

        test_player.card_states[i] = CardState.DISCARD_THEIRS_TOP
        test_player.update_card_down(i)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS


def test_discard_card(test_player: Player):
    test_player.card_states[0] = CardState.DISCARD_MINE_TOP
    for i in range(1, 52):  # Use 0 as the initial discard card
        test_player.add_card_from_deck(i)
    for i in range(1, 52):
        test_player.discard_card(i, i - 1)
        assert test_player.card_states[i] == CardState.DISCARD_MINE_TOP
        assert test_player.card_states[i - 1] == CardState.DISCARD_MINE
        assert not test_player.has_card(i)


def test_report_opponent_discarded(test_player: Player):
    test_player.card_states[0] = CardState.DISCARD_THEIRS_TOP
    for i in range(1, 52):
        test_player.report_opponent_discarded(i, i - 1)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS_TOP
        assert test_player.card_states[i - 1] == CardState.DISCARD_THEIRS


def test_add_card_from_discard(test_player: Player):
    test_player.card_states.fill(CardState.DISCARD_MINE)
    test_player.card_states[51] = CardState.DISCARD_MINE_TOP
    for i in range(51, 1, -1):
        test_player.add_card_from_discard(i, i - 1)
        assert test_player.card_states[i] == CardState.MINE_FROM_DISCARD
        assert test_player.card_states[i - 1] == CardState.DISCARD_MINE_TOP
        assert test_player.card_states[i - 2] == CardState.DISCARD_MINE
        assert test_player.has_card(i)


def test_report_opponent_drew_from_discard(test_player: Player):
    test_player.card_states.fill(CardState.DISCARD_MINE)
    test_player.card_states[51] = CardState.DISCARD_MINE_TOP
    for i in range(51, 1, -1):
        test_player.report_opponent_drew_from_discard(i, i - 1)
        assert test_player.card_states[i] == CardState.THEIRS_FROM_DISCARD
        assert test_player.card_states[i - 1] == CardState.DISCARD_MINE_TOP
        assert test_player.card_states[i - 2] == CardState.DISCARD_MINE
        assert not test_player.has_card(i)


def test_eq_cards(test_player: Player):
    for i in range(52):
        temp_player = deepcopy(test_player)
        temp_player.add_card_from_deck(i)
        assert not test_player == temp_player
        test_player.add_card_from_deck(i)
        assert test_player == temp_player
    for i in range(52):
        temp_player = deepcopy(test_player)
        temp_player.add_card_from_discard(i, 0)
        assert not test_player == temp_player
        test_player.add_card_from_discard(i, 0)
        assert test_player == temp_player
    temp_player = deepcopy(test_player)
    temp_player.score = 5
    assert not test_player == temp_player
    test_player.score = 5
    assert test_player == temp_player

# TODO ADD EMPTY DISCARD TESTS
