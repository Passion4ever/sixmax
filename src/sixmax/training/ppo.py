"""PPO algorithm implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..network import PolicyValueNetwork
from .buffer import RolloutBuffer
from .config import TrainingConfig


class PPO:
    """Proximal Policy Optimization algorithm.

    Implements PPO-Clip with value function clipping and entropy regularization.
    """

    def __init__(
        self,
        network: PolicyValueNetwork,
        config: TrainingConfig,
    ):
        """Initialize PPO.

        Args:
            network: Policy-value network to train.
            config: Training configuration.
        """
        self.network = network
        self.config = config

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
        )

    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute PPO loss.

        Args:
            batch: Batch of experiences from buffer.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        legal_masks = batch["legal_masks"]

        # Forward pass
        new_log_probs, values, entropy = self.network.evaluate_actions(
            states, legal_masks, actions
        )

        # Policy loss (PPO-Clip)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = values.squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # Compute metrics
        with torch.no_grad():
            clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
            approx_kl = ((ratio - 1) - (ratio.log())).mean()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
            "clip_fraction": clip_fraction.item(),
            "approx_kl": approx_kl.item(),
        }

        return total_loss, metrics

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Perform PPO update using collected experiences.

        Args:
            buffer: Buffer containing rollout experiences.

        Returns:
            Dictionary of averaged metrics over all updates.
        """
        all_metrics: dict[str, list[float]] = {}

        for _ in range(self.config.n_epochs):
            for batch in buffer.get_batches(self.config.batch_size):
                # Compute loss
                loss, metrics = self.compute_loss(batch)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # Average metrics
        return {key: sum(values) / len(values) for key, values in all_metrics.items()}

    def get_action(
        self,
        state: dict[str, Tensor],
        legal_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Get action from policy.

        Args:
            state: Game state batch.
            legal_mask: Legal action mask.
            deterministic: If True, use argmax instead of sampling.

        Returns:
            Tuple of (action, log_prob, value).
        """
        with torch.no_grad():
            return self.network.get_action(state, legal_mask, deterministic)

    def get_value(self, state: dict[str, Tensor]) -> Tensor:
        """Get value estimate for state.

        Args:
            state: Game state batch.

        Returns:
            Value estimate.
        """
        with torch.no_grad():
            return self.network.get_value(state)
