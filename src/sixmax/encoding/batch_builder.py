"""Batch builder for converting game states to encoder input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .action_encoder import MAX_ACTIONS, encode_action_sequence
from .card_embedding import (
    CARD_PADDING,
    RANK_PADDING,
    SUIT_PADDING,
    card_to_rank_suit,
    normalize_hole_cards,
    pad_board,
)
from .state_encoder import NUM_BOARD_CARDS, NUM_HOLE_CARDS, OPPONENT_INFO_DIM, SELF_INFO_DIM

if TYPE_CHECKING:
    from sixmax.engine import PokerGame


class StateBatchBuilder:
    """
    Build batches of encoded states from game instances.

    Converts PokerGame state to tensors suitable for StateEncoder.
    """

    def __init__(self, max_actions: int = MAX_ACTIONS):
        """
        Initialize batch builder.

        Args:
            max_actions: Maximum number of actions to encode
        """
        self.max_actions = max_actions

    def build_single(self, game: "PokerGame", player_id: int) -> dict[str, Tensor]:
        """
        Build a single sample from game state.

        Args:
            game: PokerGame instance
            player_id: Player seat to build state for

        Returns:
            Dictionary of tensors for StateEncoder
        """
        state = game.get_state_for_player(player_id)
        return self.build_from_dict(state)

    def build_from_dict(self, state: dict) -> dict[str, Tensor]:
        """
        Build a single sample from state dictionary.

        Args:
            state: State dictionary from game.get_state_for_player()

        Returns:
            Dictionary of tensors for StateEncoder
        """
        # Hole cards (normalized order)
        hole = state["hole_cards"]
        if hole is not None:
            hole = normalize_hole_cards(hole[0], hole[1])
            hole_ranks = torch.tensor(
                [card_to_rank_suit(c)[0] for c in hole], dtype=torch.long
            )
            hole_suits = torch.tensor(
                [card_to_rank_suit(c)[1] for c in hole], dtype=torch.long
            )
        else:
            hole_ranks = torch.full((NUM_HOLE_CARDS,), RANK_PADDING, dtype=torch.long)
            hole_suits = torch.full((NUM_HOLE_CARDS,), SUIT_PADDING, dtype=torch.long)

        # Board cards (padded)
        board = pad_board(state.get("board", []), NUM_BOARD_CARDS)
        board_ranks = torch.tensor(
            [card_to_rank_suit(c)[0] for c in board], dtype=torch.long
        )
        board_suits = torch.tensor(
            [card_to_rank_suit(c)[1] for c in board], dtype=torch.long
        )

        # Self info
        self_info = self._encode_self_info(state)

        # Opponent info
        opponent_info = self._encode_opponents(state.get("opponents", []))

        # Action history
        actions, action_mask = encode_action_sequence(
            state.get("actions", []), self.max_actions
        )

        return {
            "hole_ranks": hole_ranks,
            "hole_suits": hole_suits,
            "board_ranks": board_ranks,
            "board_suits": board_suits,
            "self_info": self_info,
            "opponent_info": opponent_info,
            "actions": actions,
            "action_mask": action_mask,
        }

    def build_batch(
        self, games: list["PokerGame"], player_ids: list[int]
    ) -> dict[str, Tensor]:
        """
        Build a batch from multiple games.

        Args:
            games: List of PokerGame instances
            player_ids: List of player seats (one per game)

        Returns:
            Batched dictionary of tensors
        """
        samples = [
            self.build_single(game, pid) for game, pid in zip(games, player_ids)
        ]
        return self._collate(samples)

    def build_batch_from_dicts(self, states: list[dict]) -> dict[str, Tensor]:
        """
        Build a batch from state dictionaries.

        Args:
            states: List of state dictionaries

        Returns:
            Batched dictionary of tensors
        """
        samples = [self.build_from_dict(state) for state in states]
        return self._collate(samples)

    def _collate(self, samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Collate samples into a batch."""
        return {key: torch.stack([s[key] for s in samples]) for key in samples[0].keys()}

    def _encode_self_info(self, state: dict) -> Tensor:
        """
        Encode self information → 14 dims.

        Components:
        - position: 6 dims (one-hot)
        - stack: 1 dim (normalized)
        - pot: 1 dim (normalized)
        - to_call: 1 dim (normalized)
        - street: 4 dims (one-hot)
        - raise_count: 1 dim (normalized)
        """
        features = []

        # Position one-hot (6 dims)
        position = torch.zeros(6)
        pos_value = state.get("my_position", 0)
        if isinstance(pos_value, int) and 0 <= pos_value < 6:
            position[pos_value] = 1.0
        features.append(position)

        # Numeric features (4 dims)
        stack = state.get("my_stack", 100.0) / 100.0
        pot = state.get("pot", 0.0) / 100.0
        to_call = state.get("to_call", 0.0) / 100.0
        raise_count = state.get("raise_count", 0) / 4.0

        features.append(
            torch.tensor([stack, pot, to_call, raise_count], dtype=torch.float32)
        )

        # Street one-hot (4 dims)
        street = torch.zeros(4)
        street_value = state.get("street", 0)
        if isinstance(street_value, int) and 0 <= street_value < 4:
            street[street_value] = 1.0
        features.append(street)

        result = torch.cat(features)
        assert result.shape == (SELF_INFO_DIM,), f"Expected {SELF_INFO_DIM}, got {result.shape}"
        return result

    def _encode_opponents(self, opponents: list[dict]) -> Tensor:
        """
        Encode opponent information → 15 dims.

        5 opponents × 3 features each:
        - is_active: 1 dim
        - stack: 1 dim (normalized)
        - invested: 1 dim (normalized)
        """
        features = []

        for i in range(5):
            if i < len(opponents):
                opp = opponents[i]
                features.extend(
                    [
                        1.0 if opp.get("is_active", False) else 0.0,
                        opp.get("stack", 100.0) / 100.0,
                        opp.get("invested", 0.0) / 100.0,
                    ]
                )
            else:
                # Padding for missing opponents
                features.extend([0.0, 1.0, 0.0])

        result = torch.tensor(features, dtype=torch.float32)
        assert result.shape == (OPPONENT_INFO_DIM,), f"Expected {OPPONENT_INFO_DIM}, got {result.shape}"
        return result


class VectorizedEncoder:
    """
    Vectorized encoder for batch processing.

    Combines StateBatchBuilder and StateEncoder for efficient batch encoding.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize vectorized encoder.

        Args:
            device: Device to run encoder on ('cpu' or 'cuda')
        """
        from .state_encoder import StateEncoder

        self.device = device
        self.encoder = StateEncoder().to(device)
        self.builder = StateBatchBuilder()

    @torch.no_grad()
    def encode_games(
        self, games: list["PokerGame"], player_ids: list[int]
    ) -> Tensor:
        """
        Encode multiple game states.

        Args:
            games: List of PokerGame instances
            player_ids: List of player seats

        Returns:
            (N, 261) Encoded states
        """
        batch = self.builder.build_batch(games, player_ids)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.encoder(batch)

    @torch.no_grad()
    def encode_states(self, states: list[dict]) -> Tensor:
        """
        Encode state dictionaries.

        Args:
            states: List of state dictionaries

        Returns:
            (N, 261) Encoded states
        """
        batch = self.builder.build_batch_from_dicts(states)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.encoder(batch)
