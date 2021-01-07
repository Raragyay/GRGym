import time
from timeit import default_timer

import numpy as np

from GRGym.agent import BaseAgent
from GRGym.environment import Environment
from GRGym.environment.observation import ActionPhase, PlayerID


class Match:
    SCORE_LIMIT = 100

    def __init__(self, agent_1: BaseAgent, agent_2: BaseAgent):
        self.observation_state = None
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.initialized = False
        pass

    def simulate_matches(self, count: int):
        environment = Environment()
        for match_num in range(count):
            p1_score = 0
            p2_score = 0
            draw_count = 0
            while True:  # Run games until someone exceeds 100 score
                observation = environment.reset()
                done = False
                reward = 0
                while not done:
                    player_id = observation.player_id
                    action_array = self.get_agent_to_process(player_id).act(observation.card_observations)
                    action: int = None
                    action_phase = observation.action_phase
                    if action_phase == ActionPhase.DRAW:
                        action = action_array[52] >= action_array[53]
                    elif action_phase == ActionPhase.CALL_BEFORE_DISCARD or action_phase == \
                            ActionPhase.CALL_AFTER_DISCARD:
                        action = action_array[54] >= action_array[55]
                    else:
                        card_states = np.array(observation.card_observations, dtype=np.int8)
                        action = self.get_card_to_discard(action_array, (card_states == 1) | (card_states == 2))
                    observation, done, reward = environment.step(action)
                if reward == 0:
                    draw_count += 1
                elif observation.player_id == 1:
                    if reward > 0:
                        p1_score += reward
                    else:
                        p2_score -= reward  # Add the positive, which means inverting the negative
                elif observation.player_id == 2:
                    if reward > 0:
                        p2_score += reward
                    else:
                        p1_score -= reward  # Add the positive, which means inverting the negative
                self.agent_1.reset()
                self.agent_2.reset()
                if p1_score >= self.SCORE_LIMIT or p2_score >= self.SCORE_LIMIT:
                    print(f"There were {draw_count} draws.")
                    break

    def get_agent_to_process(self, agent_id) -> BaseAgent:
        if agent_id == 1:  # TODO refactor to enum
            return self.agent_1
        elif agent_id == 2:
            return self.agent_2
        else:
            raise ValueError(f"Agent ID {agent_id} is not a valid id.")

    @staticmethod
    def get_card_to_discard(action: np.ndarray, player_card_mask: np.ndarray):
        # print(action)
        # print(player_card_mask)
        # print(player_card_mask[30])
        # Sort the cards by how much the player prefers them. Negative makes it descending.
        # Keep it this way for benchmarking, convert to first method later.
        # actions_sorted = np.argsort(action[0:52])[::-1]
        actions_sorted = np.argsort(-action[0:52])
        # Sort the mask booleans (0 or 1 depending on if the player has them) by how much the player prefers them.
        card_mask_sorted = player_card_mask[actions_sorted]
        # Only take cards that the player has in their hand.
        cards_in_hand_sorted = actions_sorted[card_mask_sorted]
        # Take the most preferred card in the player's hand.
        card_to_discard = cards_in_hand_sorted[0]
        return card_to_discard
