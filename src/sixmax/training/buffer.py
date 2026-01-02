"""Rollout buffer for storing trajectories and computing GAE."""

from __future__ import annotations

import torch
from torch import Tensor


class RolloutBuffer:
    """Buffer for storing rollout experiences and computing advantages.

    Stores trajectories from self-play and computes Generalized Advantage
    Estimation (GAE) for PPO updates.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        """Initialize the rollout buffer.

        Args:
            gamma: Discount factor for rewards.
            gae_lambda: Lambda parameter for GAE.
            device: Device to store tensors on.
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Clear all stored experiences."""
        self.states: list[dict[str, Tensor]] = []
        self.actions: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.values: list[Tensor] = []
        self.log_probs: list[Tensor] = []
        self.dones: list[Tensor] = []
        self.legal_masks: list[Tensor] = []

        # Computed after rollout
        self.advantages: Tensor | None = None
        self.returns: Tensor | None = None

    def add(
        self,
        state: dict[str, Tensor],
        action: Tensor,
        reward: float,
        value: Tensor,
        log_prob: Tensor,
        done: bool,
        legal_mask: Tensor,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            state: Game state batch dict from StateBatchBuilder.
            action: Action taken (scalar tensor).
            reward: Reward received.
            value: Value estimate V(s).
            log_prob: Log probability of action.
            done: Whether episode (hand) ended.
            legal_mask: Legal action mask.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(torch.tensor(reward, device=self.device))
        self.values.append(value.detach())
        self.log_probs.append(log_prob.detach())
        self.dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))
        self.legal_masks.append(legal_mask)

    def compute_returns_and_advantages(self, last_value: Tensor) -> None:
        """Compute GAE advantages and returns.

        Args:
            last_value: Value estimate for the last state (for bootstrapping).
        """
        n = len(self.rewards)
        if n == 0:
            self.advantages = torch.tensor([], device=self.device)
            self.returns = torch.tensor([], device=self.device)
            return

        # Stack tensors
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values).squeeze(-1)  # Keep at least 1D
        dones = torch.stack(self.dones)

        # Ensure values is 1D
        if values.dim() == 0:
            values = values.unsqueeze(0)

        # Append last value for bootstrapping
        last_val = last_value.detach().view(-1)[0:1]  # Ensure 1D single element
        values_with_last = torch.cat([values, last_val])

        # Compute GAE
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_value = values_with_last[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values_with_last[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            )

        self.advantages = advantages
        self.returns = advantages + values

    def get_batches(self, batch_size: int):
        """Generate mini-batches for training.

        Args:
            batch_size: Size of each mini-batch.

        Yields:
            Dictionary containing batch data.
        """
        n = len(self.rewards)
        if n == 0:
            return

        # Stack all data
        actions = torch.stack(self.actions)
        log_probs = torch.stack(self.log_probs)
        legal_masks = torch.stack(self.legal_masks)

        assert self.advantages is not None
        assert self.returns is not None

        # Normalize advantages
        advantages = self.advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate random indices
        indices = torch.randperm(n, device=self.device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            # Gather batch states
            batch_states = self._gather_states(batch_indices)

            yield {
                "states": batch_states,
                "actions": actions[batch_indices],
                "old_log_probs": log_probs[batch_indices],
                "advantages": advantages[batch_indices],
                "returns": self.returns[batch_indices],
                "legal_masks": legal_masks[batch_indices],
            }

    def _gather_states(self, indices: Tensor) -> dict[str, Tensor]:
        """Gather states at given indices.

        Args:
            indices: Indices to gather.

        Returns:
            Batched state dictionary.
        """
        batch_states: dict[str, list[Tensor]] = {}

        for idx in indices.tolist():
            state = self.states[idx]
            for key, value in state.items():
                if key not in batch_states:
                    batch_states[key] = []
                batch_states[key].append(value)

        return {key: torch.stack(values) for key, values in batch_states.items()}

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.rewards)
