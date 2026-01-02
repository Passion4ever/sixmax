"""Card embedding module for encoding poker cards."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Padding values
RANK_PADDING = 13  # For cards with no rank (padding)
SUIT_PADDING = 4  # For cards with no suit (padding)
CARD_PADDING = 52  # Card value for padding


class CardEmbedding(nn.Module):
    """
    Embedding layer for poker cards.

    Each card is represented by two embeddings:
    - Rank embedding: 13 ranks (2-A) + 1 padding = 14 values
    - Suit embedding: 4 suits (♠♥♦♣) + 1 padding = 5 values

    Output dimension: rank_dim + suit_dim (default 16 + 8 = 24)
    """

    def __init__(self, rank_dim: int = 16, suit_dim: int = 8):
        """
        Initialize card embedding.

        Args:
            rank_dim: Dimension of rank embedding (default 16)
            suit_dim: Dimension of suit embedding (default 8)
        """
        super().__init__()
        self.rank_dim = rank_dim
        self.suit_dim = suit_dim
        self.output_dim = rank_dim + suit_dim

        # Embeddings
        self.rank_embed = nn.Embedding(14, rank_dim)  # 13 ranks + padding
        self.suit_embed = nn.Embedding(5, suit_dim)  # 4 suits + padding

        # Initialize with small random values
        nn.init.normal_(self.rank_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.suit_embed.weight, mean=0.0, std=0.02)

    def forward(self, ranks: Tensor, suits: Tensor) -> Tensor:
        """
        Encode cards to embeddings.

        Args:
            ranks: (batch, num_cards) Card ranks (0-12, or 13 for padding)
            suits: (batch, num_cards) Card suits (0-3, or 4 for padding)

        Returns:
            (batch, num_cards, output_dim) Card embeddings
        """
        rank_emb = self.rank_embed(ranks)  # (batch, num_cards, rank_dim)
        suit_emb = self.suit_embed(suits)  # (batch, num_cards, suit_dim)
        return torch.cat([rank_emb, suit_emb], dim=-1)  # (batch, num_cards, output_dim)

    def forward_flat(self, ranks: Tensor, suits: Tensor) -> Tensor:
        """
        Encode cards and flatten to single vector.

        Args:
            ranks: (batch, num_cards) Card ranks
            suits: (batch, num_cards) Card suits

        Returns:
            (batch, num_cards * output_dim) Flattened card embeddings
        """
        emb = self.forward(ranks, suits)
        return emb.view(emb.size(0), -1)


def card_to_rank_suit(card: int) -> tuple[int, int]:
    """
    Convert card value to (rank, suit).

    Args:
        card: Card value 0-51, or 52 for padding

    Returns:
        (rank, suit) tuple
    """
    if card >= 52:
        return RANK_PADDING, SUIT_PADDING
    return card // 4, card % 4


def normalize_hole_cards(card1: int, card2: int) -> tuple[int, int]:
    """
    Normalize hole cards order (higher rank first).

    This ensures A♠K♥ and K♥A♠ produce the same encoding.

    Args:
        card1: First card value
        card2: Second card value

    Returns:
        (high_card, low_card) ordered by rank
    """
    rank1, rank2 = card1 // 4, card2 // 4
    if rank1 >= rank2:
        return card1, card2
    return card2, card1


def pad_board(board: list[int], max_len: int = 5) -> list[int]:
    """
    Pad board cards to fixed length.

    Args:
        board: List of card values (0-5 cards)
        max_len: Target length (default 5)

    Returns:
        Padded list of length max_len
    """
    padded = list(board)
    while len(padded) < max_len:
        padded.append(CARD_PADDING)
    return padded[:max_len]


def cards_to_tensors(cards: list[int]) -> tuple[Tensor, Tensor]:
    """
    Convert list of card values to rank and suit tensors.

    Args:
        cards: List of card values

    Returns:
        (ranks, suits) tensors
    """
    ranks = []
    suits = []
    for card in cards:
        r, s = card_to_rank_suit(card)
        ranks.append(r)
        suits.append(s)
    return torch.tensor(ranks, dtype=torch.long), torch.tensor(suits, dtype=torch.long)
