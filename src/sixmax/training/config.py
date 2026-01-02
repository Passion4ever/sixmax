"""Training configuration for PPO."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    """PPO training hyperparameters.

    Attributes:
        learning_rate: Adam optimizer learning rate.
        clip_epsilon: PPO clipping range.
        value_coef: Value loss coefficient.
        entropy_coef: Entropy regularization coefficient.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        n_epochs: Number of epochs per update.
        batch_size: Mini-batch size for updates.
        max_grad_norm: Gradient clipping norm.
        n_games: Number of parallel games.
        n_hands_per_update: Hands to collect before each update.
        total_hands: Total hands to train.
        log_interval: Hands between logging.
        save_interval: Hands between checkpoints.
        checkpoint_dir: Directory for saving checkpoints.
        wandb_project: Weights & Biases project name.
        wandb_run_name: Optional run name for wandb.
        device: Device to use ('cuda', 'mps', 'cpu', or 'auto').
        seed: Random seed for reproducibility.
    """

    # PPO hyperparameters
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 4
    batch_size: int = 4096
    max_grad_norm: float = 0.5

    # Training configuration
    n_games: int = 16384
    n_hands_per_update: int = 2000
    total_hands: int = 10_000_000

    # Logging and checkpoints
    log_interval: int = 1000
    save_interval: int = 10000
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "6max-poker"
    wandb_run_name: str | None = None

    # Device and reproducibility
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "clip_epsilon": self.clip_epsilon,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "n_games": self.n_games,
            "n_hands_per_update": self.n_hands_per_update,
            "total_hands": self.total_hands,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "device": self.device,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SmallConfig(TrainingConfig):
    """Small configuration for testing on Mac/CPU.

    Also known as TestConfig (alias provided for convenience).
    Runs ~1 minute on Mac CPU.
    """

    n_games: int = 64
    n_hands_per_update: int = 500
    total_hands: int = 20000
    batch_size: int = 128
    log_interval: int = 2000
    save_interval: int = 10000
    device: str = "cpu"


# Alias for backward compatibility
TestConfig = SmallConfig
