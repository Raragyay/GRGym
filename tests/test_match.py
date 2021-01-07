import pytest

from GRGym.agent import RandomAgent
from GRGym.simulation.match import Match
from GRGym.agent import HandBuiltAgent


def test_simulate():
    agent_1 = HandBuiltAgent()
    agent_2 = HandBuiltAgent()
    m = Match(agent_1, agent_2)
    m.simulate_matches(1)
