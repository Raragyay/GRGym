import pytest

from GRGym.agent import BaseAgent
from GRGym.environment.environment import Environment
from GRGym.environment.player import Player
from .cyconftest import player_with_cards as cy_player_with_cards


def base_agent_class():
    return BaseAgent


def base_env_class():
    return Environment  # TODO change to testing agent maybe?


def base_player_class():
    return Player


@pytest.fixture
def test_env():
    return base_env_class()(base_agent_class()())


@pytest.fixture
def test_player():
    return base_player_class()()


@pytest.fixture
def player_with_cards(test_player):
    return cy_player_with_cards(test_player)
