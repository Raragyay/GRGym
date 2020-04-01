from typing import Dict, List

import numpy as np

from player import Player


class Environment:

    def __init__(self, opponent_agent):
        self.card_states = np.zeros((52,), np.int8)
        self.player_1 = Player()
        self.player_2 = Player()
        self.deck = np.arange(52)
        self.discard_pile: List[int] = []

        self.opponents: Dict[Player, Player] = {
            self.player_1: self.player_2,
            self.player_2: self.player_1}

        self.draw_phase = True  # Either draw phase or discard phase

    def reset(self) -> np.ndarray:
        self.card_states = np.zeros((52,), np.int8)
        self.player_1.reset()
        self.player_2.reset()
        self.deck = np.arange(52)
        np.random.shuffle(self.deck)
        self.discard_pile = []

        self.draw_from_deck(self.player_1, 10)
        self.draw_from_deck(self.player_2, 10)
        self.draw_phase = True
        return self.build_observations(self.player_1)

    def step(self, action: np.ndarray) -> (np.ndarray, int):
        """
        Step through one interaction with the environment.
        :param action:
        A 56 length vector. First 52 elements correspond to the 52 cards.
        Next 2 elements correspond to draw from deck and draw from discard
        Last 2 elements correspond to knock and not knock
        :return:
        A tuple of results.
        The first element is the new observation vector.
        The second element is the reward.
        If the player won the hand, this is set to 1. If they won the match, this is set to 2.
        If they lost the hand, this is set to -1. If they lost the match, this is set to -2.
        Otherwise, this is set to 0.

        """
        if self.draw_phase:
            if action[52] >= action[53]:
                self.draw_from_deck(self.player_1)
            else:
                self.draw_from_discard(self.player_1)
        else:  # Discard phase
            card_to_discard = self.get_card_to_discard(action)
            self.discard_card(self.player_1, card_to_discard)
            if action[54] >= action[55]:  # Player wants to knock

                pass
                # TODO knock

            # TODO run opponent draw and discard
        self.draw_phase = not self.draw_phase
        # TODO build observations

    def get_card_to_discard(self, action):
        # Sort the cards by how much the player prefers them. Negative makes it descending.
        actions_sorted = np.argsort(-action[0:52])
        # Sort the mask booleans (0 or 1 depending on if the player has them) by how much the player prefers them.
        card_mask_sorted = self.player_1.hand_mask()[actions_sorted]
        # Only take cards that the player has in their hand.
        cards_in_hand_sorted = actions_sorted[card_mask_sorted]
        # Take the most preferred card in the player's hand.
        card_to_discard = cards_in_hand_sorted[0]
        return card_to_discard

    def draw_from_deck(self, player: Player, num_of_cards: int = 1):
        assert player in self.opponents
        assert len(self.deck) >= num_of_cards
        for card_val in self.deck[:num_of_cards]:
            player.add_card_from_deck(card_val)
        self.deck = self.deck[num_of_cards:]
        return

    def draw_from_discard(self, player: Player):
        drawn_card: int = self.discard_pile.pop()
        new_top_discard: int = self.discard_pile[-1]
        player.add_card_from_discard(drawn_card, new_top_discard)
        self.opponents[player].report_opponent_drew_from_discard(drawn_card, new_top_discard)

    def discard_card(self, player: Player, card_to_discard: int):
        previous_top = self.discard_pile[-1]
        self.discard_pile.append(card_to_discard)
        player.discard_card(card_to_discard, previous_top)
        self.opponents[player].report_opponent_discarded(card_to_discard, previous_top)

    def build_observations(self, player: Player) -> np.ndarray:
        """
        Builds observation array.
        52 elements for card_state
        1 element for current player score
        1 element for opponent score
        1 element for number of cards still left in deck
        1 element for whether or not the player is drawing
        1 element for whether or not the player is discarding
        Total of 57
        :param player:
        :return:
        """
        observation: np.ndarray = np.empty(57, np.int8)
        observation[0:52] = player.card_states
        observation[52] = player.score
        observation[53] = self.opponents[player].score
        observation[54] = len(self.deck)
        observation[55] = self.draw_phase
        observation[56] = not self.draw_phase
        return observation


if __name__ == '__main__':
    np.random.seed(1)
    env = Environment(Player())
    env.reset()
    print(env.player_1.hand.__repr__())
    print(env.player_2.hand.__repr__())
    fake_action = np.zeros(56)
    print(env.player_1.has_card(2))
    fake_action[2] = 2
    fake_action[51] = 1
    fake_action[52] = 1
    env.step(fake_action)
    print(env.player_1.hand.__repr__())
    env.step(fake_action)
    print(env.player_1.hand.__repr__())
