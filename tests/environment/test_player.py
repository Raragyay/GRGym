from copy import deepcopy

import numpy as np
import pytest

from GRGym.environment.card_state import CardState
from GRGym.environment.player import Player
from .cytest_player import *


@pytest.fixture
def test_player():
    return cytest_player()


def test_reset(test_player: Player):
    cytest_reset(test_player)


def test_add_card_from_deck(test_player: Player):
    cytest_add_card_from_deck(test_player)


def test_has_card(test_player: Player):
    cytest_has_card(test_player)


def test_card_list(test_player: Player):
    cytest_card_list(test_player)


def test_hand_mask(test_player: Player):
    cytest_hand_mask(test_player)


def test_update_card_to_top(test_player: Player):
    cytest_update_card_to_top(test_player)


def test_update_card_down(test_player: Player):
    cytest_update_card_down(test_player)


def test_discard_card(test_player: Player):
    cytest_discard_card(test_player)


def test_report_opponent_discarded(test_player: Player):
    cytest_report_opponent_discarded(test_player)


def test_add_card_from_discard(test_player: Player):
    cytest_add_card_from_discard(test_player)


def test_report_opponent_drew_from_discard(test_player: Player):
    cytest_report_opponent_drew_from_discard(test_player)


def test_eq_cards(test_player: Player):
    cytest_eq(test_player)

# TODO ADD EMPTY DISCARD TESTS
