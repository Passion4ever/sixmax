"""Main training loop for PPO self-play."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ..network import PolicyValueNetwork, create_policy_value_network
from .buffer import RolloutBuffer
from .config import TrainingConfig
from .ppo import PPO
from .rollout import ParallelSelfPlay


class Trainer:
    """PPO trainer with self-play.

    Orchestrates the training loop:
    1. Collect self-play experiences
    2. Update policy with PPO
    3. Log metrics and save checkpoints
    """

    def __init__(
        self,
        config: TrainingConfig,
        network: PolicyValueNetwork | None = None,
        wandb_enabled: bool = False,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration.
            network: Optional pre-created network. If None, creates new one.
            wandb_enabled: Whether to log to Weights & Biases.
        """
        self.config = config
        self.wandb_enabled = wandb_enabled

        # Resolve device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        # Create or use provided network
        if network is None:
            self.network = create_policy_value_network().to(self.device)
        else:
            self.network = network.to(self.device)

        # Create PPO algorithm
        self.ppo = PPO(self.network, config)

        # Create self-play collector
        self.self_play = ParallelSelfPlay(
            network=self.network,
            n_games=config.n_games,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=self.device,
            seed=config.seed,
        )

        # Training state
        self.total_hands = 0
        self.total_updates = 0
        self.start_time: float | None = None

        # Checkpoint directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(config.checkpoint_dir) / timestamp
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if enabled
        self._wandb_run = None
        if wandb_enabled:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.to_dict(),
            )
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            self.wandb_enabled = False

    def train(self) -> dict[str, Any]:
        """Run full training loop.

        Returns:
            Final training metrics.
        """
        self.start_time = time.time()

        print(f"Starting training on {self.device}")
        print(f"  Total hands: {self.config.total_hands:,}")
        print(f"  Parallel games: {self.self_play.total_games}")
        print(f"  Hands per update: {self.config.n_hands_per_update}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print()

        metrics_history: list[dict[str, float]] = []

        while self.total_hands < self.config.total_hands:
            # Collect self-play data
            hands_per_collector = max(
                1, self.config.n_hands_per_update // len(self.self_play.collectors)
            )
            buffer = self.self_play.collect(hands_per_collector, show_progress=True)
            hands_collected = len(buffer)
            self.total_hands += hands_collected

            # Update policy
            update_metrics = self.ppo.update(buffer)
            self.total_updates += 1

            # Add training info to metrics
            elapsed = time.time() - self.start_time
            update_metrics.update({
                "hands": self.total_hands,
                "updates": self.total_updates,
                "elapsed_time": elapsed,
                "hands_per_second": self.total_hands / elapsed if elapsed > 0 else 0,
                "buffer_size": hands_collected,
            })

            metrics_history.append(update_metrics)

            # Log progress
            if self.total_hands % self.config.log_interval < hands_collected:
                self._log_metrics(update_metrics)

            # Save checkpoint
            if self.total_hands % self.config.save_interval < hands_collected:
                self.save_checkpoint()

        # Final save
        self.save_checkpoint(final=True)

        # Close wandb
        if self._wandb_run is not None:
            self._wandb_run.finish()

        return self._compute_final_metrics(metrics_history)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to console and wandb."""
        # Console output
        elapsed = metrics.get("elapsed_time", 0)
        hands = metrics.get("hands", 0)
        hps = metrics.get("hands_per_second", 0)

        print(
            f"[{elapsed:6.1f}s] Hands: {hands:>8,} | "
            f"HPS: {hps:,.0f} | "
            f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
            f"Value Loss: {metrics.get('value_loss', 0):.4f} | "
            f"Entropy: {metrics.get('entropy', 0):.4f}"
        )

        # Wandb logging
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=self.total_hands)

    def save_checkpoint(self, final: bool = False) -> str:
        """Save training checkpoint.

        Args:
            final: Whether this is the final checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        if final:
            filename = "final.pt"
        else:
            filename = f"checkpoint_{self.total_hands:08d}.pt"

        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "total_hands": self.total_hands,
            "total_updates": self.total_updates,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        return str(path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_hands = checkpoint.get("total_hands", 0)
        self.total_updates = checkpoint.get("total_updates", 0)

        print(f"Loaded checkpoint: {path}")
        print(f"  Resuming from {self.total_hands:,} hands")

    def _compute_final_metrics(
        self, history: list[dict[str, float]]
    ) -> dict[str, Any]:
        """Compute final training metrics.

        Args:
            history: List of metrics from each update.

        Returns:
            Summary statistics.
        """
        if not history:
            return {}

        # Average over last 10% of training
        n_recent = max(1, len(history) // 10)
        recent = history[-n_recent:]

        def avg(key: str) -> float:
            values = [m[key] for m in recent if key in m]
            return sum(values) / len(values) if values else 0.0

        total_time = time.time() - self.start_time if self.start_time else 0

        return {
            "total_hands": self.total_hands,
            "total_updates": self.total_updates,
            "total_time_seconds": total_time,
            "final_policy_loss": avg("policy_loss"),
            "final_value_loss": avg("value_loss"),
            "final_entropy": avg("entropy"),
            "average_hands_per_second": self.total_hands / total_time if total_time > 0 else 0,
        }


def train_from_config(config: TrainingConfig, wandb_enabled: bool = False) -> dict[str, Any]:
    """Convenience function to train from config.

    Args:
        config: Training configuration.
        wandb_enabled: Whether to enable wandb logging.

    Returns:
        Final training metrics.
    """
    trainer = Trainer(config, wandb_enabled=wandb_enabled)
    return trainer.train()
