#!/usr/bin/env python3
"""Training script for 6-max poker AI.

Usage:
    # Local testing on Mac
    uv run scripts/train.py --config test

    # Full training on GPU
    uv run scripts/train.py --config full --device cuda --wandb

    # Resume from checkpoint
    uv run scripts/train.py --resume checkpoints/checkpoint_00100000.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sixmax.training import SmallConfig, TrainingConfig, Trainer, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 6-max poker AI with PPO self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config preset
    parser.add_argument(
        "--config",
        type=str,
        default="test",
        choices=["test", "small", "medium", "full"],
        help="Configuration preset to use",
    )

    # Override individual settings
    parser.add_argument("--total-hands", type=int, help="Total hands to train")
    parser.add_argument("--n-games", type=int, help="Number of parallel games")
    parser.add_argument("--batch-size", type=int, help="Mini-batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device (cpu, cuda, mps, auto)")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--save-interval", type=int, help="Hands between checkpoints")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--run-name", type=str, help="Run name for wandb")
    parser.add_argument("--log-interval", type=int, help="Hands between logging")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def get_config_preset(name: str) -> TrainingConfig:
    """Get configuration preset by name."""
    if name == "test":
        return SmallConfig()
    elif name == "small":
        return TrainingConfig(
            n_games=256,
            total_hands=100_000,
            batch_size=512,
            log_interval=1000,
            save_interval=10000,
        )
    elif name == "medium":
        return TrainingConfig(
            n_games=4096,
            total_hands=1_000_000,
            batch_size=2048,
            log_interval=5000,
            save_interval=50000,
        )
    elif name == "full":
        return TrainingConfig()  # Default full config
    else:
        raise ValueError(f"Unknown config preset: {name}")


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Get base config
    config = get_config_preset(args.config)

    # Apply overrides
    if args.total_hands is not None:
        config.total_hands = args.total_hands
    if args.n_games is not None:
        config.n_games = args.n_games
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.run_name is not None:
        config.wandb_run_name = args.run_name
    config.seed = args.seed

    # Set random seed
    set_seed(config.seed)

    # Print config
    print("=" * 60)
    print("6-Max Poker AI Training")
    print("=" * 60)
    print()
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print()

    # Create trainer
    trainer = Trainer(config, wandb_enabled=args.wandb)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        metrics = trainer.train()

        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print()
        print("Final Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
