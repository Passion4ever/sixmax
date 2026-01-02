"""Tests for the state encoding module."""

import pytest
import torch

from sixmax.encoding import (
    ActionHistoryEncoder,
    CardEmbedding,
    StateEncoder,
    StateBatchBuilder,
    card_to_rank_suit,
    create_empty_batch,
    encode_action_sequence,
    encode_single_action,
    normalize_hole_cards,
    pad_board,
)
from sixmax.engine import ActionType, PokerGame, Street


class TestCardEmbedding:
    """Test card embedding module."""

    def test_embedding_shape(self):
        """Test embedding output shape."""
        embed = CardEmbedding(rank_dim=16, suit_dim=8)

        ranks = torch.tensor([[0, 12]])  # 2 and A
        suits = torch.tensor([[0, 1]])  # spade and heart

        output = embed(ranks, suits)
        assert output.shape == (1, 2, 24)

    def test_embedding_batch(self):
        """Test embedding with batch."""
        embed = CardEmbedding()

        batch_size = 4
        ranks = torch.randint(0, 13, (batch_size, 5))
        suits = torch.randint(0, 4, (batch_size, 5))

        output = embed(ranks, suits)
        assert output.shape == (batch_size, 5, 24)

    def test_card_to_rank_suit(self):
        """Test card value conversion."""
        # Ace of spades (card 48)
        rank, suit = card_to_rank_suit(48)
        assert rank == 12  # Ace
        assert suit == 0  # Spade

        # 2 of clubs (card 3)
        rank, suit = card_to_rank_suit(3)
        assert rank == 0  # 2
        assert suit == 3  # Club

        # Padding card
        rank, suit = card_to_rank_suit(52)
        assert rank == 13
        assert suit == 4

    def test_normalize_hole_cards(self):
        """Test hole card normalization."""
        # As (48), Kh (45) -> should stay same order
        c1, c2 = normalize_hole_cards(48, 45)
        assert c1 // 4 >= c2 // 4

        # Kh (45), As (48) -> should be swapped
        c1, c2 = normalize_hole_cards(45, 48)
        assert c1 // 4 >= c2 // 4
        assert c1 == 48

    def test_pad_board(self):
        """Test board padding."""
        # Flop (3 cards)
        board = [0, 4, 8]
        padded = pad_board(board)
        assert len(padded) == 5
        assert padded[:3] == [0, 4, 8]
        assert padded[3:] == [52, 52]

        # River (5 cards)
        board = [0, 4, 8, 12, 16]
        padded = pad_board(board)
        assert len(padded) == 5
        assert padded == board


class TestActionEncoder:
    """Test action history encoder."""

    def test_single_action_encoding(self):
        """Test single action encoding."""
        token = encode_single_action(
            player=0,
            action=ActionType.RAISE,
            amount=2.5,
            street=Street.PREFLOP,
        )
        assert token.shape == (17,)

        # Check one-hot encoding
        assert token[0] == 1.0  # player 0
        assert token[1:6].sum() == 0.0  # other players
        assert token[6 + ActionType.RAISE] == 1.0  # action type

    def test_action_sequence_encoding(self):
        """Test action sequence encoding."""
        from sixmax.engine.state import ActionRecord

        actions = [
            ActionRecord(player=0, action=ActionType.RAISE, amount=2.5, street=Street.PREFLOP),
            ActionRecord(player=1, action=ActionType.FOLD, amount=0, street=Street.PREFLOP),
        ]

        encoded, mask = encode_action_sequence(actions, max_len=24)

        assert encoded.shape == (24, 17)
        assert mask.shape == (24,)
        assert mask[:2].sum() == 0  # First 2 are real
        assert mask[2:].sum() == 22  # Rest are padding

    def test_action_history_encoder(self):
        """Test transformer action encoder."""
        encoder = ActionHistoryEncoder(hidden_dim=64)

        batch_size = 4
        seq_len = 10
        actions = torch.randn(batch_size, seq_len, 17)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, 5:] = True  # Last 5 are padding

        output = encoder(actions, mask)
        assert output.shape == (batch_size, 64)


class TestStateEncoder:
    """Test main state encoder."""

    def test_encoder_output_dim(self):
        """Test encoder output dimension is 261."""
        encoder = StateEncoder()
        assert encoder.get_output_dim() == 261

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = StateEncoder()
        batch = create_empty_batch(4)

        output = encoder(batch)
        assert output.shape == (4, 261)

    def test_encoder_with_real_data(self):
        """Test encoder with realistic data."""
        encoder = StateEncoder()

        batch = {
            "hole_ranks": torch.tensor([[12, 11], [10, 9]]),  # AK, QJ
            "hole_suits": torch.tensor([[0, 1], [2, 3]]),
            "board_ranks": torch.tensor([[0, 4, 8, 13, 13], [2, 6, 10, 12, 13]]),
            "board_suits": torch.tensor([[0, 0, 0, 4, 4], [1, 1, 1, 1, 4]]),
            "self_info": torch.randn(2, 14),
            "opponent_info": torch.randn(2, 15),
            "actions": torch.randn(2, 24, 17),
            "action_mask": torch.zeros(2, 24, dtype=torch.bool),
        }

        output = encoder(batch)
        assert output.shape == (2, 261)
        assert not torch.isnan(output).any()


class TestStateBatchBuilder:
    """Test batch builder integration."""

    def test_build_from_game(self):
        """Test building batch from PokerGame."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        builder = StateBatchBuilder()
        sample = builder.build_single(game, player_id=0)

        assert sample["hole_ranks"].shape == (2,)
        assert sample["hole_suits"].shape == (2,)
        assert sample["board_ranks"].shape == (5,)
        assert sample["self_info"].shape == (14,)
        assert sample["opponent_info"].shape == (15,)

    def test_build_batch(self):
        """Test building batch from multiple games."""
        games = []
        for i in range(4):
            game = PokerGame()
            game.seed(42 + i)
            game.reset_hand()
            games.append(game)

        builder = StateBatchBuilder()
        batch = builder.build_batch(games, player_ids=[0, 1, 2, 3])

        assert batch["hole_ranks"].shape == (4, 2)
        assert batch["self_info"].shape == (4, 14)

    def test_full_pipeline(self):
        """Test full encoding pipeline."""
        # Create game
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Play a few actions
        game.step(ActionType.RAISE)  # UTG raises
        game.step(ActionType.FOLD)  # HJ folds

        # Build and encode
        builder = StateBatchBuilder()
        encoder = StateEncoder()

        sample = builder.build_single(game, player_id=2)  # CO perspective
        batch = {k: v.unsqueeze(0) for k, v in sample.items()}

        output = encoder(batch)
        assert output.shape == (1, 261)

    def test_postflop_encoding(self):
        """Test encoding in postflop situation."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Get to flop by everyone calling
        done = False
        for _ in range(20):
            if done or game.get_state().hand_over:
                break

            legal = game.get_legal_actions()
            state = game.get_state()

            if state.street != Street.PREFLOP:
                break

            if legal[ActionType.CHECK_CALL]:
                _, _, done = game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                _, _, done = game.step(ActionType.FOLD)

        # Now we should have board cards
        state = game.get_state()
        if state.street.value > 0:
            builder = StateBatchBuilder()
            sample = builder.build_single(game, player_id=0)

            # Board should not be all padding
            assert (sample["board_ranks"] != 13).any()


class TestIntegration:
    """Integration tests."""

    def test_encoding_determinism(self):
        """Test that encoding is deterministic."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        builder = StateBatchBuilder()
        encoder = StateEncoder()
        encoder.eval()

        sample1 = builder.build_single(game, player_id=0)
        sample2 = builder.build_single(game, player_id=0)

        batch1 = {k: v.unsqueeze(0) for k, v in sample1.items()}
        batch2 = {k: v.unsqueeze(0) for k, v in sample2.items()}

        with torch.no_grad():
            out1 = encoder(batch1)
            out2 = encoder(batch2)

        assert torch.allclose(out1, out2)

    def test_different_players_different_encodings(self):
        """Test that different players get different encodings."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        builder = StateBatchBuilder()
        encoder = StateEncoder()
        encoder.eval()

        sample0 = builder.build_single(game, player_id=0)
        sample1 = builder.build_single(game, player_id=1)

        batch0 = {k: v.unsqueeze(0) for k, v in sample0.items()}
        batch1 = {k: v.unsqueeze(0) for k, v in sample1.items()}

        with torch.no_grad():
            out0 = encoder(batch0)
            out1 = encoder(batch1)

        # Different players should have different encodings
        assert not torch.allclose(out0, out1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_action_sequence(self):
        """Test encoding with no actions."""
        encoded, mask = encode_action_sequence([], max_len=24)
        assert encoded.shape == (24, 17)
        assert mask.shape == (24,)
        assert mask.all()  # All should be padding

    def test_full_action_sequence(self):
        """Test encoding with maximum actions."""
        from sixmax.engine.state import ActionRecord

        actions = [
            ActionRecord(player=i % 6, action=ActionType.CHECK_CALL, amount=0, street=Street.PREFLOP)
            for i in range(30)  # More than max
        ]

        encoded, mask = encode_action_sequence(actions, max_len=24)
        assert encoded.shape == (24, 17)
        assert not mask.any()  # No padding since we have more actions than max

    def test_action_encoder_all_masked(self):
        """Test action encoder with all tokens masked."""
        encoder = ActionHistoryEncoder(hidden_dim=64)
        encoder.eval()

        batch_size = 4
        seq_len = 10
        actions = torch.zeros(batch_size, seq_len, 17)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)  # All masked

        with torch.no_grad():
            output = encoder(actions, mask)

        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
        # All masked should return zeros
        assert torch.allclose(output, torch.zeros_like(output))

    def test_action_encoder_partial_masked(self):
        """Test action encoder with partial masking."""
        encoder = ActionHistoryEncoder(hidden_dim=64)
        encoder.eval()

        batch_size = 2
        seq_len = 10
        actions = torch.randn(batch_size, seq_len, 17)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # First sample: half masked
        mask[0, 5:] = True
        # Second sample: all masked
        mask[1, :] = True

        with torch.no_grad():
            output = encoder(actions, mask)

        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
        # Second sample should be zeros (all masked)
        assert torch.allclose(output[1], torch.zeros(64))

    def test_cards_to_tensors(self):
        """Test batch card conversion."""
        from sixmax.encoding import cards_to_tensors

        cards = [0, 4, 8, 52, 52]  # 3 real cards + 2 padding
        ranks, suits = cards_to_tensors(cards)

        assert ranks.shape == (5,)
        assert suits.shape == (5,)
        assert ranks[-1] == 13  # Padding rank
        assert suits[-1] == 4  # Padding suit

    def test_vectorized_encoder(self):
        """Test VectorizedEncoder convenience class."""
        from sixmax.encoding import VectorizedEncoder

        vec_encoder = VectorizedEncoder(device="cpu")

        games = []
        for i in range(4):
            game = PokerGame()
            game.seed(42 + i)
            game.reset_hand()
            games.append(game)

        output = vec_encoder.encode_games(games, player_ids=[0, 1, 2, 3])
        assert output.shape == (4, 261)
        assert not torch.isnan(output).any()

    def test_vectorized_encoder_from_dicts(self):
        """Test VectorizedEncoder with state dicts."""
        from sixmax.encoding import VectorizedEncoder

        vec_encoder = VectorizedEncoder(device="cpu")

        # Create state dicts
        states = []
        for i in range(3):
            game = PokerGame()
            game.seed(42 + i)
            game.reset_hand()
            state_dict = game.get_state_for_player(i % 6)
            states.append(state_dict)

        output = vec_encoder.encode_states(states)
        assert output.shape == (3, 261)
        assert not torch.isnan(output).any()

    def test_build_batch_from_dicts(self):
        """Test building batch from state dictionaries."""
        builder = StateBatchBuilder()

        states = []
        for i in range(2):
            game = PokerGame()
            game.seed(42 + i)
            game.reset_hand()
            states.append(game.get_state_for_player(0))

        batch = builder.build_batch_from_dicts(states)

        assert batch["hole_ranks"].shape == (2, 2)
        assert batch["self_info"].shape == (2, 14)

    def test_missing_hole_cards(self):
        """Test encoding when hole cards are None."""
        builder = StateBatchBuilder()

        state = {
            "hole_cards": None,
            "board": [],
            "my_position": 0,
            "my_stack": 100.0,
            "pot": 1.5,
            "to_call": 1.0,
            "raise_count": 0,
            "street": 0,
            "opponents": [],
            "actions": [],
        }

        sample = builder.build_from_dict(state)

        # Should use padding values
        assert sample["hole_ranks"].shape == (2,)
        assert (sample["hole_ranks"] == 13).all()  # Padding rank
        assert (sample["hole_suits"] == 4).all()  # Padding suit

    def test_action_encoder_no_mask(self):
        """Test action encoder without mask."""
        encoder = ActionHistoryEncoder(hidden_dim=64)
        encoder.eval()

        batch_size = 2
        seq_len = 5
        actions = torch.randn(batch_size, seq_len, 17)

        with torch.no_grad():
            output = encoder(actions, mask=None)

        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
