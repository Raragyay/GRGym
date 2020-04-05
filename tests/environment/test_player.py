import numpy as np
import pytest

from environment.card_state import CardState
from environment.player import Player


def test_reset():
    test_player = Player()
    np.testing.assert_array_equal(test_player.hand_mask(), np.zeros(52))
    np.testing.assert_array_equal(test_player.card_states, np.zeros(52))
    test_player.add_card_from_deck(23)
    test_player.add_card_from_deck(42)
    test_player.add_card_from_deck(1)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(test_player.hand_mask(), np.zeros(52))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(test_player.card_states, np.zeros(52))
    test_player.reset()
    np.testing.assert_array_equal(test_player.hand_mask(), np.zeros(52))
    np.testing.assert_array_equal(test_player.card_states, np.zeros(52))


def test_add_card_from_deck():
    test_player = Player()
    for i in range(52):
        assert test_player.card_states[i] == CardState.UNKNOWN
        test_player.add_card_from_deck(i)
        assert test_player.card_states[i] == CardState.MINE_FROM_DECK
        assert i in test_player.card_list()
        test_player.add_card_from_deck(i)
        assert test_player.card_states[i] == CardState.MINE_FROM_DECK


def test_has_card():
    test_player = Player()
    for i in range(52):
        assert not test_player.has_card(i)
        test_player.add_card_from_deck(i)
        assert test_player.has_card(i)


def test_card_list():
    test_player = Player()
    cards_to_add = [2, 6, 23, 51, 11, 0, 45, 26, 32]
    for i in range(len(cards_to_add)):
        test_player.add_card_from_deck(cards_to_add[i])
        np.testing.assert_array_equal(test_player.card_list(), np.fromiter(sorted(cards_to_add[:i + 1]), np.int8))


def test_hand_mask():
    test_player = Player()
    cards_to_add = [2, 6, 23, 51, 11, 0, 45, 26, 32]
    model_mask = np.zeros(52, np.bool)
    for i in range(len(cards_to_add)):
        test_player.add_card_from_deck(cards_to_add[i])
        model_mask[cards_to_add[i]] = True
        np.testing.assert_array_equal(test_player.hand_mask(), model_mask)


def test_update_card_to_top():
    test_player = Player()
    for i in range(52):
        test_player.card_states[i] = CardState.DISCARD_MINE
        test_player.update_card_to_top(i)
        assert test_player.card_states[i] == CardState.DISCARD_MINE_TOP

        test_player.card_states[i] = CardState.DISCARD_THEIRS
        test_player.update_card_to_top(i)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS_TOP


def test_update_card_down():
    test_player = Player()
    for i in range(52):
        test_player.card_states[i] = CardState.DISCARD_MINE_TOP
        test_player.update_card_down(i)
        assert test_player.card_states[i] == CardState.DISCARD_MINE

        test_player.card_states[i] = CardState.DISCARD_THEIRS_TOP
        test_player.update_card_down(i)
        assert test_player.card_states[i] == CardState.DISCARD_THEIRS
