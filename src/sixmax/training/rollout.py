"""Self-play data collection for PPO training with vectorized inference."""

from __future__ import annotations

import random
import sys
import time
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

    Uses vectorized inference for efficient GPU utilization.
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

    def collect_rollout(self, n_hands: int, show_progress: bool = False) -> RolloutBuffer:
        """Collect experiences from n_hands of self-play using vectorized inference.

        Args:
            n_hands: Number of hands to play across all games.
            show_progress: Whether to show progress output.

        Returns:
            Buffer containing collected experiences.
        """
        self.buffer.reset()

        hands_completed = 0
        steps_collected = 0
        start_time = time.time()
        last_progress_time = start_time

        # Start all games
        for game in self.games:
            game.reset_hand()

        while hands_completed < n_hands:
            # Step 1: Collect all games needing decisions
            pending_games = []
            pending_indices = []
            pending_states = []
            pending_players = []
            pending_legal_masks = []

            for idx, game in enumerate(self.games):
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

                # Collect state for batch processing
                state_dict = game.get_state_for_player(current_player)
                pending_games.append(game)
                pending_indices.append(idx)
                pending_states.append(state_dict)
                pending_players.append(current_player)
                pending_legal_masks.append(legal_mask)

            # No games need decisions
            if not pending_games:
                break

            # Step 2: Batch inference for all pending games
            batch_size = len(pending_games)

            # Build batched state tensors
            state_batch = self.batch_builder.build_batch_from_dicts(pending_states)
            state_batch = {k: v.to(self.device) for k, v in state_batch.items()}

            # Build batched legal mask tensor
            legal_mask_tensor = torch.tensor(
                pending_legal_masks, dtype=torch.bool, device=self.device
            )

            # Batch forward pass through network
            with torch.no_grad():
                actions, log_probs, values = self.network.get_action(
                    state_batch, legal_mask_tensor, deterministic=False
                )

            # Step 3: Execute actions and store experiences
            for i in range(batch_size):
                game = pending_games[i]
                current_player = pending_players[i]
                action_int = actions[i].item()

                # Store single state tensors for buffer
                single_state = {k: v[i] for k, v in state_batch.items()}
                single_legal_mask = legal_mask_tensor[i]

                # Execute action
                _, rewards, done = game.step(ActionType(action_int))

                # Get reward for current player
                reward = rewards.get(current_player, 0.0) if done else 0.0

                # Store experience
                self.buffer.add(
                    state=single_state,
                    action=actions[i],
                    reward=reward,
                    value=values[i],
                    log_prob=log_probs[i],
                    done=done,
                    legal_mask=single_legal_mask,
                )

                steps_collected += 1

                if done:
                    hands_completed += 1

            # Progress output
            if show_progress:
                current_time = time.time()
                if current_time - last_progress_time >= 5.0:  # Every 5 seconds
                    elapsed = current_time - start_time
                    hps = hands_completed / elapsed if elapsed > 0 else 0
                    sps = steps_collected / elapsed if elapsed > 0 else 0
                    print(
                        f"  [Collecting] Hands: {hands_completed}/{n_hands} "
                        f"({100*hands_completed/n_hands:.1f}%) | "
                        f"HPS: {hps:.0f} | Steps/s: {sps:.0f}",
                        flush=True
                    )
                    last_progress_time = current_time

        # Compute returns and advantages
        # Use zero as last value since all episodes are complete
        last_value = torch.zeros(1, device=self.device)
        self.buffer.compute_returns_and_advantages(last_value)

        return self.buffer

    def collect_steps(self, n_steps: int) -> RolloutBuffer:
        """Collect a fixed number of decision steps using vectorized inference.

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
            # Step 1: Collect all games needing decisions
            pending_games = []
            pending_states = []
            pending_players = []
            pending_legal_masks = []

            for game in self.games:
                if steps_collected + len(pending_games) >= n_steps:
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

                # Collect for batch
                state_dict = game.get_state_for_player(current_player)
                pending_games.append(game)
                pending_states.append(state_dict)
                pending_players.append(current_player)
                pending_legal_masks.append(legal_mask)

            if not pending_games:
                break

            # Step 2: Batch inference
            batch_size = len(pending_games)

            state_batch = self.batch_builder.build_batch_from_dicts(pending_states)
            state_batch = {k: v.to(self.device) for k, v in state_batch.items()}

            legal_mask_tensor = torch.tensor(
                pending_legal_masks, dtype=torch.bool, device=self.device
            )

            with torch.no_grad():
                actions, log_probs, values = self.network.get_action(
                    state_batch, legal_mask_tensor, deterministic=False
                )

            # Step 3: Execute and store
            for i in range(batch_size):
                if steps_collected >= n_steps:
                    break

                game = pending_games[i]
                current_player = pending_players[i]
                action_int = actions[i].item()

                single_state = {k: v[i] for k, v in state_batch.items()}
                single_legal_mask = legal_mask_tensor[i]

                _, rewards, done = game.step(ActionType(action_int))
                reward = rewards.get(current_player, 0.0)

                self.buffer.add(
                    state=single_state,
                    action=actions[i],
                    reward=reward,
                    value=values[i],
                    log_prob=log_probs[i],
                    done=done,
                    legal_mask=single_legal_mask,
                )

                steps_collected += 1

        # Bootstrap value for last state
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
        games_per_collector: int = 1024,
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

    def collect(self, n_hands_per_collector: int, show_progress: bool = True) -> RolloutBuffer:
        """Collect experiences from all collectors.

        Args:
            n_hands_per_collector: Hands to collect per collector.
            show_progress: Whether to show progress output.

        Returns:
            Aggregated buffer with all experiences.
        """
        start_time = time.time()
        total_hands = n_hands_per_collector * len(self.collectors)

        if show_progress:
            print(f"  Collecting {total_hands} hands from {len(self.collectors)} collectors...")

        # Collect from all collectors
        buffers = []
        for i, collector in enumerate(self.collectors):
            if show_progress and len(self.collectors) > 1:
                print(f"    Collector {i+1}/{len(self.collectors)}...", end=" ", flush=True)

            buf = collector.collect_rollout(n_hands_per_collector, show_progress=False)
            buffers.append(buf)

            if show_progress and len(self.collectors) > 1:
                print(f"done ({len(buf)} transitions)")

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

        elapsed = time.time() - start_time
        if show_progress:
            hps = total_hands / elapsed if elapsed > 0 else 0
            print(f"  Collection complete: {len(merged)} transitions in {elapsed:.1f}s ({hps:.0f} hands/s)")

        return merged

    @property
    def total_games(self) -> int:
        """Total number of parallel games across all collectors."""
        return sum(c.n_games for c in self.collectors)
