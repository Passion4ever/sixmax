"""Action history encoder using Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Action encoding dimensions
ACTION_TOKEN_DIM = 17  # player(6) + action(6) + amount(1) + street(4)
MAX_ACTIONS = 24  # Maximum actions per hand


class ActionHistoryEncoder(nn.Module):
    """
    Encode action history sequence using Transformer.

    Input: Sequence of action tokens (batch, seq_len, 17)
    Output: Fixed-size representation (batch, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int = ACTION_TOKEN_DIM,
        hidden_dim: int = 64,
        num_heads: int = 4,
        max_len: int = MAX_ACTIONS,
        dropout: float = 0.1,
    ):
        """
        Initialize action history encoder.

        Args:
            input_dim: Dimension of each action token (default 17)
            hidden_dim: Hidden dimension of transformer (default 64)
            num_heads: Number of attention heads (default 4)
            max_len: Maximum sequence length (default 24)
            dropout: Dropout rate (default 0.1)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )

        # Layer norm for output
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, actions: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Encode action sequence.

        Args:
            actions: (batch, seq_len, input_dim) Action token sequence
            mask: (batch, seq_len) Padding mask (True = padding, will be ignored)

        Returns:
            (batch, hidden_dim) Encoded action history
        """
        batch_size, seq_len = actions.shape[:2]

        # Handle empty sequences
        if seq_len == 0:
            return torch.zeros(batch_size, self.hidden_dim, device=actions.device)

        # Handle all-padding sequences (no real actions)
        # This prevents NaN from Transformer when all tokens are masked
        if mask is not None:
            # Check which samples have all tokens masked
            all_masked = mask.all(dim=1)  # (batch,)
            if all_masked.all():
                # All samples are fully masked, return zeros
                return torch.zeros(batch_size, self.hidden_dim, device=actions.device)

        # Project to hidden dimension
        x = self.input_proj(actions)  # (batch, seq_len, hidden_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :seq_len, :]

        # Apply transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling over non-padding tokens
        if mask is not None:
            # Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, 1)
            mask_expanded = (~mask).unsqueeze(-1).float()
            # Sum of valid tokens
            x_sum = (x * mask_expanded).sum(dim=1)
            # Count of valid tokens (avoid division by zero)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            x = x_sum / count

            # Handle samples where all tokens were masked (set to zeros)
            all_masked = mask.all(dim=1)  # (batch,)
            if all_masked.any():
                x = x.clone()
                x[all_masked] = 0.0
        else:
            x = x.mean(dim=1)

        # Output normalization
        x = self.output_norm(x)

        return x  # (batch, hidden_dim)


def encode_single_action(
    player: int,
    action: int,
    amount: float,
    street: int,
    num_players: int = 6,
    num_actions: int = 6,
) -> Tensor:
    """
    Encode a single action to a 17-dimensional tensor.

    Args:
        player: Player seat (0-5)
        action: Action type (0-5)
        amount: Bet/raise amount in BB
        street: Street (0-3)
        num_players: Number of players (default 6)
        num_actions: Number of action types (default 6)

    Returns:
        (17,) Action token tensor
    """
    features = []

    # Player one-hot (6 dims)
    player_onehot = torch.zeros(num_players)
    player_onehot[player] = 1.0
    features.append(player_onehot)

    # Action type one-hot (6 dims)
    action_onehot = torch.zeros(num_actions)
    action_onehot[action] = 1.0
    features.append(action_onehot)

    # Amount normalized (1 dim)
    features.append(torch.tensor([amount / 100.0]))

    # Street one-hot (4 dims)
    street_onehot = torch.zeros(4)
    street_onehot[street] = 1.0
    features.append(street_onehot)

    return torch.cat(features)  # (17,)


def encode_action_sequence(
    actions: list,
    max_len: int = MAX_ACTIONS,
) -> tuple[Tensor, Tensor]:
    """
    Encode a list of actions to padded sequence.

    Args:
        actions: List of ActionRecord objects
        max_len: Maximum sequence length

    Returns:
        (actions_tensor, mask_tensor)
        - actions_tensor: (max_len, 17)
        - mask_tensor: (max_len,) True for padding positions
    """
    encoded = []

    for action in actions[:max_len]:
        token = encode_single_action(
            player=action.player,
            action=int(action.action),
            amount=action.amount,
            street=int(action.street),
        )
        encoded.append(token)

    # Pad to max_len
    while len(encoded) < max_len:
        encoded.append(torch.zeros(ACTION_TOKEN_DIM))

    actions_tensor = torch.stack(encoded)  # (max_len, 17)

    # Create mask (True for padding)
    mask = torch.zeros(max_len, dtype=torch.bool)
    num_real = min(len(actions), max_len)
    mask[num_real:] = True

    return actions_tensor, mask
