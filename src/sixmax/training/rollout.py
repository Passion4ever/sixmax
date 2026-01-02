"""Self-play data collection for PPO training."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..encoding import StateBatchBuilder
from ..engine import NUM_ACTIONS, ActionType, PokerGame
from .buffer import RolloutBuffer

if TYPE_CHECKING:
    from ..network import PolicyValueNetwork


class SelfPlayCollector:
    """Collect self-play experiences for PPO training.

    Runs multiple parallel games with all players controlled by the same
    policy network, collecting transitions for training.
    """

    def __init__(
        self,
        network: PolicyValueNetwork,
        n_games: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
        seed: int | None = None,
    ):
        """Initialize self-play collector.

        Args:
            network: Policy-value network for action selection.
            n_games: Number of parallel games.
            gamma: Discount factor for GAE.
            gae_lambda: Lambda for GAE computation.
            device: Device for tensors.
            seed: Random seed for reproducibility.
        """
        self.network = network
        self.n_games = n_games
        self.device = device
        self.batch_builder = StateBatchBuilder()

        # Initialize games
        self.games = [PokerGame() for _ in range(n_games)]
        if seed is not None:
            for i, game in enumerate(self.games):
                game.seed(seed + i)

        # Initialize buffers (one per player per game)
        # We collect from all 6 players in each game
        self.buffer = RolloutBuffer(gamma, gae_lambda, device)

        # Track active hands
        self._rng = random.Random(seed)

    def collect_rollout(self, n_hands: int) -> RolloutBuffer:
        """Collect experiences from n_hands of self-play.

        Args:
            n_hands: Number of hands to play across all games.

        Returns:
            Buffer containing collected experiences.
        """
        self.buffer.reset()

        hands_completed = 0

        # Start all games
        for game in self.games:
            game.reset_hand()

        while hands_completed < n_hands:
            made_progress = False

            # Process all games that have a decision point
            for game in self.games:
                if hands_completed >= n_hands:
                    break

                state = game.get_state()

                # Reset hand if over
                if state.hand_over:
                    game.reset_hand()
                    state = game.get_state()

                current_player = state.current_player
                legal_mask = game.get_legal_actions()

                # Skip if no legal actions (shouldn't happen)
                if not any(legal_mask):
                    continue

                made_progress = True

                # Build state for current player
                state_dict = game.get_state_for_player(current_player)
                state_batch = self.batch_builder.build_from_dict(state_dict)
                state_batch = {k: v.unsqueeze(0).to(self.device) for k, v in state_batch.items()}

                # Get legal mask tensor
                legal_mask_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=self.device)

                # Get action from network
                with torch.no_grad():
                    action, log_prob, value = self.network.get_action(
                        state_batch, legal_mask_tensor.unsqueeze(0), deterministic=False
                    )

                # Execute action first to get reward
                action_int = action.item()
                _, rewards, done = game.step(ActionType(action_int))

                # Get reward for current player
                reward = rewards.get(current_player, 0.0) if done else 0.0

                # Store experience
                self.buffer.add(
                    state={k: v.squeeze(0) for k, v in state_batch.items()},
                    action=action.squeeze(0),
                    reward=reward,
                    value=value.squeeze(0),
                    log_prob=log_prob.squeeze(0),
                    done=done,
                    legal_mask=legal_mask_tensor,
                )

                if done:
                    hands_completed += 1

            # Safety check to avoid infinite loop
            if not made_progress:
                break

        # Compute returns and advantages
        # Use zero as last value since all episodes are complete
        last_value = torch.zeros(1, device=self.device)
        self.buffer.compute_returns_and_advantages(last_value)

        return self.buffer

    def collect_steps(self, n_steps: int) -> RolloutBuffer:
        """Collect a fixed number of decision steps.

        Alternative to collect_rollout when you want a fixed number
        of transitions rather than complete hands.

        Args:
            n_steps: Number of decision steps to collect.

        Returns:
            Buffer containing collected experiences.
        """
        self.buffer.reset()

        steps_collected = 0

        # Start all games
        for game in self.games:
            if game.state is None or game.state.hand_over:
                game.reset_hand()

        while steps_collected < n_steps:
            for game in self.games:
                if steps_collected >= n_steps:
                    break

                state = game.get_state()

                # Reset if hand is over
                if state.hand_over:
                    game.reset_hand()
                    state = game.get_state()

                current_player = state.current_player
                legal_mask = game.get_legal_actions()

                if not any(legal_mask):
                    continue

                # Build state
                state_dict = game.get_state_for_player(current_player)
                state_batch = self.batch_builder.build_from_dict(state_dict)
                state_batch = {k: v.unsqueeze(0).to(self.device) for k, v in state_batch.items()}

                legal_mask_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=self.device)

                # Get action
                with torch.no_grad():
                    action, log_prob, value = self.network.get_action(
                        state_batch, legal_mask_tensor.unsqueeze(0), deterministic=False
                    )

                # Execute action
                action_int = action.item()
                _, rewards, done = game.step(ActionType(action_int))

                # Get reward for current player
                reward = rewards.get(current_player, 0.0)

                # Store experience
                self.buffer.add(
                    state={k: v.squeeze(0) for k, v in state_batch.items()},
                    action=action.squeeze(0),
                    reward=reward,
                    value=value.squeeze(0),
                    log_prob=log_prob.squeeze(0),
                    done=done,
                    legal_mask=legal_mask_tensor,
                )

                steps_collected += 1

        # Bootstrap value for last state
        # Get value for current state of first non-done game
        last_value = torch.zeros(1, device=self.device)
        for game in self.games:
            state = game.get_state()
            if not state.hand_over:
                current_player = state.current_player
                state_dict = game.get_state_for_player(current_player)
                state_batch = self.batch_builder.build_from_dict(state_dict)
                state_batch = {k: v.unsqueeze(0).to(self.device) for k, v in state_batch.items()}

                with torch.no_grad():
                    last_value = self.network.get_value(state_batch).squeeze()
                break

        self.buffer.compute_returns_and_advantages(last_value)

        return self.buffer


class ParallelSelfPlay:
    """Higher-level parallel self-play manager.

    Manages multiple SelfPlayCollector instances and aggregates
    their experiences.
    """

    def __init__(
        self,
        network: PolicyValueNetwork,
        n_games: int = 16384,
        games_per_collector: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
        seed: int | None = None,
    ):
        """Initialize parallel self-play.

        Args:
            network: Shared policy-value network.
            n_games: Total number of parallel games.
            games_per_collector: Games per collector (for batching).
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            device: Device for tensors.
            seed: Random seed.
        """
        self.network = network
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Create collectors
        n_collectors = max(1, n_games // games_per_collector)
        self.collectors = [
            SelfPlayCollector(
                network=network,
                n_games=min(games_per_collector, n_games - i * games_per_collector),
                gamma=gamma,
                gae_lambda=gae_lambda,
                device=device,
                seed=seed + i * games_per_collector if seed else None,
            )
            for i in range(n_collectors)
            if i * games_per_collector < n_games
        ]

    def collect(self, n_hands_per_collector: int) -> RolloutBuffer:
        """Collect experiences from all collectors.

        Args:
            n_hands_per_collector: Hands to collect per collector.

        Returns:
            Aggregated buffer with all experiences.
        """
        # Collect from all collectors
        buffers = [c.collect_rollout(n_hands_per_collector) for c in self.collectors]

        # Merge buffers
        merged = RolloutBuffer(self.gamma, self.gae_lambda, self.device)

        for buf in buffers:
            merged.states.extend(buf.states)
            merged.actions.extend(buf.actions)
            merged.rewards.extend(buf.rewards)
            merged.values.extend(buf.values)
            merged.log_probs.extend(buf.log_probs)
            merged.dones.extend(buf.dones)
            merged.legal_masks.extend(buf.legal_masks)

        # Compute returns for merged buffer
        if len(merged.rewards) > 0:
            last_value = torch.zeros(1, device=self.device)
            merged.compute_returns_and_advantages(last_value)

        return merged

    @property
    def total_games(self) -> int:
        """Total number of parallel games across all collectors."""
        return sum(c.n_games for c in self.collectors)
