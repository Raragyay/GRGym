import logging
import typing
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from agent.baseagent import BaseAgent
from deadwood.deadwood_counter_dp import DeadwoodCounter
from environment.action_result import ActionResult
from environment.player import Player
from meld.run import Run
from meld.set import Set


class Environment:
    SCORE_LIMIT = 100

    def __init__(self, opponent_agent: BaseAgent):
        self.card_states = np.zeros((52,), np.int8)
        self.player_1 = Player()
        self.player_2 = Player()
        self.opponent_agent = opponent_agent
        self.deck = np.arange(52)
        self.discard_pile: List[int] = []

        self.opponents: Dict[Player, Player] = {
            self.player_1: self.player_2,
            self.player_2: self.player_1}

        self.draw_phase = True  # Either draw phase or discard phase

    def reset_hand(self) -> np.ndarray:
        """
        Resets the environment in preparation for the next hand. Player scores are not reset.

        :return:
        """
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int]:
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
        action_result = 0
        if self.draw_phase:
            self.run_draw(action, self.player_1)
        else:  # Discard phase
            action_result = self.run_discard(action, self.player_1, is_player_1=True)
        self.draw_phase = not self.draw_phase
        if action_result != ActionResult.NO_CHANGE:
            self.reset_hand()
        return self.build_observations(self.player_1), action_result

    def run_draw(self, action: np.ndarray, player: Player):
        if self.wants_to_draw_from_deck(action):
            self.draw_from_deck(player)
        else:
            self.draw_from_discard(player)

    def run_discard(self, action: np.ndarray, player: Player, is_player_1: bool) -> int:
        if self.wants_to_knock(action) and self.is_gin(player):
            return self.score_big_gin(player)
        card_to_discard = self.get_card_to_discard(action, player)
        self.discard_card(player, card_to_discard)
        if self.wants_to_knock(action) and self.can_knock(player):
            if self.is_gin(player):
                return self.score_gin(player)
            else:
                score_delta = self.try_to_knock(player)
                if score_delta > 0:
                    return self.update_score(player, score_delta)
                else:
                    # Apply negative to return  relative to current player
                    return -self.update_score(self.opponents[player], -score_delta)
        if not is_player_1:
            return 0  # Avoid player 2 recursing back into player 1, want to only recurse once
        # Run player 2 draw and discard actions
        opponent_draw_action = self.opponent_agent.act(self.build_observations(self.player_2))
        self.run_draw(opponent_draw_action, self.player_2)
        opponent_discard_action = self.opponent_agent.act(self.build_observations(self.player_2))
        return self.run_discard(opponent_discard_action, self.player_2, False)

    @staticmethod
    def get_card_to_discard(action: np.ndarray, player: Player):
        # Sort the cards by how much the player prefers them. Negative makes it descending.
        actions_sorted = np.argsort(-action[0:52])
        # Sort the mask booleans (0 or 1 depending on if the player has them) by how much the player prefers them.
        card_mask_sorted = player.hand_mask()[actions_sorted]
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

    def discard_card(self, player: Player, card_to_discard: int):
        previous_top = self.discard_pile[-1]
        self.discard_pile.append(card_to_discard)
        player.discard_card(card_to_discard, previous_top)
        self.opponents[player].report_opponent_discarded(card_to_discard, previous_top)

    def try_to_knock(self, player: Player) -> int:
        deadwood_counter = DeadwoodCounter(player.card_list())
        knocking_player_deadwood = deadwood_counter.deadwood()
        knocking_player_melds = deadwood_counter.melds()
        deadwood_counter = DeadwoodCounter(self.opponents[player].card_list())
        opponent_deadwood = deadwood_counter.deadwood()
        opponent_remaining_cards = set(deadwood_counter.remaining_cards())

        # layoff into runs first. This is because any cards laid off into runs will never block layoffs into sets,
        # while the opposite could happen.
        # Example: The knocking player has a run from 3-5 of diamonds and a set of 2 of Clubs, 2 of Hearts,
        # and 2 of Spades. The opponent has deadwood remaining cards of Ace of Diamonds and 2 of Diamonds.
        # If set layoffs run first, then 2 of Diamonds would be laid off, and the Ace of Diamonds would remain.
        # However, if run layoffs run first, then both the 2 of Diamonds and the Ace of Diamonds would be laid off.

        # try to layoff into runs
        run_melds: typing.Set[Run] = {deepcopy(meld) for meld in knocking_player_melds if isinstance(meld, Run)}

        while True:
            connectable_card_to_run: Dict[int, Run] = {card: run for run in run_melds for card in
                                                       run.connectable_cards()}
            connectable_remaining_cards = opponent_remaining_cards & connectable_card_to_run.keys()
            if not connectable_remaining_cards:
                break
            for card in connectable_remaining_cards:
                connectable_run = connectable_card_to_run[card]
                if card == connectable_run.start - 1:  # Append to the left
                    connectable_run.start -= 1
                elif card == connectable_run.end + 1:
                    connectable_run.end += 1
                else:
                    logging.error(f"Card {card} was inserted into connectable_remaining_cards, but it is not an "
                                  f"extension of any run. ")
                opponent_remaining_cards.remove(card)
                opponent_deadwood -= DeadwoodCounter.deadwood_val(card)

        # try to layoff into sets
        set_melds: typing.Set[Set] = {deepcopy(meld) for meld in knocking_player_melds if isinstance(meld, Set)}
        while True:
            connectable_card_to_set: Dict[int, Set] = {card: m_set for m_set in set_melds for card in
                                                       m_set.connectable_cards()}
            connectable_remaining_cards = opponent_remaining_cards & connectable_card_to_set.keys()
            if not connectable_remaining_cards:
                break
            for card in connectable_remaining_cards:
                opponent_remaining_cards.remove(card)
                opponent_deadwood -= DeadwoodCounter.deadwood_val(card)

        # Check score difference
        if knocking_player_deadwood < opponent_deadwood:  # won hand
            return opponent_deadwood - knocking_player_deadwood
        else:  # undercut
            return -25 - (knocking_player_deadwood - opponent_deadwood)

    def score_big_gin(self, player: Player) -> ActionResult:
        score_delta = DeadwoodCounter(self.opponents[player].card_list()).deadwood() + 31
        return self.update_score(player, score_delta)

    def score_gin(self, player):
        score_delta = DeadwoodCounter(self.opponents[player].card_list()).deadwood() + 25
        return self.update_score(player, score_delta)

    def update_score(self, player: Player, score_delta: int) -> ActionResult:
        player.score += score_delta
        if player.score >= self.SCORE_LIMIT:
            return ActionResult.WON_MATCH
        else:
            return ActionResult.WON_HAND

    @staticmethod
    def wants_to_draw_from_deck(action: np.ndarray) -> bool:
        return action[52] >= action[53]

    @staticmethod
    def wants_to_knock(action: np.ndarray) -> bool:
        return action[54] >= action[55]

    @staticmethod
    def is_gin(player: Player) -> bool:
        return DeadwoodCounter(player.card_list()).deadwood() == 0

    @staticmethod
    def can_knock(player: Player) -> bool:
        return DeadwoodCounter(player.card_list()).deadwood() <= 10
