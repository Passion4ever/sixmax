"""Main state encoder module."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .action_encoder import ActionHistoryEncoder, MAX_ACTIONS
from .card_embedding import CardEmbedding

# Encoding dimensions
SELF_INFO_DIM = 14  # position(6) + stack(1) + pot(1) + to_call(1) + street(4) + raise_count(1)
OPPONENT_INFO_DIM = 15  # 5 opponents × 3 features each
NUM_HOLE_CARDS = 2
NUM_BOARD_CARDS = 5


class StateEncoder(nn.Module):
    """
    State encoder for poker game states.

    Encodes game state into a fixed-size vector for the neural network.

    Input components:
    - Hole cards: 2 cards → 48 dims (2 × 24)
    - Board cards: 5 cards → 120 dims (5 × 24)
    - Self info: position, stack, pot, etc. → 14 dims
    - Opponent info: 5 opponents × 3 features → 15 dims
    - Action history: Transformer encoded → 64 dims

    Total output: 261 dimensions
    """

    def __init__(
        self,
        rank_embed_dim: int = 16,
        suit_embed_dim: int = 8,
        action_hidden_dim: int = 64,
        max_actions: int = MAX_ACTIONS,
    ):
        """
        Initialize state encoder.

        Args:
            rank_embed_dim: Dimension of rank embedding (default 16)
            suit_embed_dim: Dimension of suit embedding (default 8)
            action_hidden_dim: Hidden dimension for action encoder (default 64)
            max_actions: Maximum number of actions to encode (default 24)
        """
        super().__init__()

        self.card_dim = rank_embed_dim + suit_embed_dim  # 24
        self.action_hidden_dim = action_hidden_dim
        self.max_actions = max_actions

        # Card embedding
        self.card_embed = CardEmbedding(rank_embed_dim, suit_embed_dim)

        # Action history encoder
        self.action_encoder = ActionHistoryEncoder(
            input_dim=17,
            hidden_dim=action_hidden_dim,
            max_len=max_actions,
        )

        # Calculate output dimension
        self.hole_dim = NUM_HOLE_CARDS * self.card_dim  # 2 × 24 = 48
        self.board_dim = NUM_BOARD_CARDS * self.card_dim  # 5 × 24 = 120
        self.output_dim = (
            self.hole_dim  # 48
            + self.board_dim  # 120
            + SELF_INFO_DIM  # 14
            + OPPONENT_INFO_DIM  # 15
            + action_hidden_dim  # 64
        )  # Total: 261

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Encode batch of game states.

        Args:
            batch: Dictionary containing:
                - hole_ranks: (B, 2) Hole card ranks
                - hole_suits: (B, 2) Hole card suits
                - board_ranks: (B, 5) Board card ranks
                - board_suits: (B, 5) Board card suits
                - self_info: (B, 14) Self information
                - opponent_info: (B, 15) Opponent information
                - actions: (B, max_actions, 17) Action sequence
                - action_mask: (B, max_actions) Padding mask

        Returns:
            (B, 261) Encoded state vectors
        """
        features = []

        # 1. Hole cards embedding → 48 dims
        hole_emb = self.card_embed(batch["hole_ranks"], batch["hole_suits"])
        hole_flat = hole_emb.view(hole_emb.size(0), -1)
        features.append(hole_flat)

        # 2. Board cards embedding → 120 dims
        board_emb = self.card_embed(batch["board_ranks"], batch["board_suits"])
        board_flat = board_emb.view(board_emb.size(0), -1)
        features.append(board_flat)

        # 3. Self info → 14 dims (already encoded)
        features.append(batch["self_info"])

        # 4. Opponent info → 15 dims (already encoded)
        features.append(batch["opponent_info"])

        # 5. Action history → 64 dims
        action_encoding = self.action_encoder(
            batch["actions"],
            batch.get("action_mask"),
        )
        features.append(action_encoding)

        # Concatenate all features
        return torch.cat(features, dim=-1)  # (B, 261)

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


def create_empty_batch(batch_size: int, device: str = "cpu") -> dict[str, Tensor]:
    """
    Create an empty batch for testing.

    Args:
        batch_size: Number of samples
        device: Device to create tensors on

    Returns:
        Empty batch dictionary
    """
    return {
        "hole_ranks": torch.zeros(batch_size, NUM_HOLE_CARDS, dtype=torch.long, device=device),
        "hole_suits": torch.zeros(batch_size, NUM_HOLE_CARDS, dtype=torch.long, device=device),
        "board_ranks": torch.full(
            (batch_size, NUM_BOARD_CARDS), 13, dtype=torch.long, device=device
        ),  # padding
        "board_suits": torch.full(
            (batch_size, NUM_BOARD_CARDS), 4, dtype=torch.long, device=device
        ),  # padding
        "self_info": torch.zeros(batch_size, SELF_INFO_DIM, device=device),
        "opponent_info": torch.zeros(batch_size, OPPONENT_INFO_DIM, device=device),
        "actions": torch.zeros(batch_size, MAX_ACTIONS, 17, device=device),
        "action_mask": torch.ones(batch_size, MAX_ACTIONS, dtype=torch.bool, device=device),
    }
