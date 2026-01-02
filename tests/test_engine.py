"""Tests for the poker game engine."""

import pytest

from sixmax.engine import (
    ActionType,
    Card,
    HandEvaluator,
    HandRank,
    PokerGame,
    Position,
    Street,
    evaluate_hand,
)


class TestCard:
    """Test Card class."""

    def test_card_creation(self):
        """Test card creation from value."""
        # Ace of spades: rank=12, suit=0
        card = Card(48)
        assert card.rank == 12
        assert card.suit == 0
        assert str(card) == "As"

    def test_card_from_string(self):
        """Test card creation from string."""
        card = Card.from_string("Kh")
        assert card.rank == 11  # King
        assert card.suit == 1  # Hearts
        assert str(card) == "Kh"

    def test_all_cards(self):
        """Test all 52 cards."""
        cards = [Card(i) for i in range(52)]
        assert len(cards) == 52
        assert len(set(c.value for c in cards)) == 52


class TestHandEvaluator:
    """Test hand evaluation."""

    def setup_method(self):
        """Set up evaluator."""
        self.evaluator = HandEvaluator()

    def _cards(self, *names: str) -> list[int]:
        """Convert card names to integers."""
        return [Card.from_string(n).value for n in names]

    def test_high_card(self):
        """Test high card detection."""
        hole = (Card.from_string("Ah").value, Card.from_string("Kd").value)
        board = self._cards("2c", "5h", "8s", "Ts", "Jc")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.HIGH_CARD

    def test_one_pair(self):
        """Test one pair detection."""
        hole = (Card.from_string("Ah").value, Card.from_string("Ad").value)
        board = self._cards("2c", "5h", "8s", "Ts", "Jc")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.ONE_PAIR

    def test_two_pair(self):
        """Test two pair detection."""
        hole = (Card.from_string("Ah").value, Card.from_string("Kd").value)
        board = self._cards("Ac", "Kh", "8s", "2s", "3c")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.TWO_PAIR

    def test_three_of_a_kind(self):
        """Test three of a kind detection."""
        hole = (Card.from_string("Ah").value, Card.from_string("Ad").value)
        board = self._cards("Ac", "5h", "8s", "Ts", "Jc")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.THREE_OF_A_KIND

    def test_straight(self):
        """Test straight detection."""
        hole = (Card.from_string("9h").value, Card.from_string("8d").value)
        board = self._cards("Tc", "Jh", "Qs", "2s", "3c")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.STRAIGHT

    def test_flush(self):
        """Test flush detection."""
        hole = (Card.from_string("Ah").value, Card.from_string("Kh").value)
        board = self._cards("2h", "5h", "8h", "Ts", "Jc")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.FLUSH

    def test_full_house(self):
        """Test full house detection."""
        hole = (Card.from_string("Ah").value, Card.from_string("Ad").value)
        board = self._cards("Ac", "Kh", "Ks", "2s", "3c")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.FULL_HOUSE

    def test_four_of_a_kind(self):
        """Test four of a kind detection."""
        hole = (Card.from_string("Ah").value, Card.from_string("Ad").value)
        board = self._cards("Ac", "As", "Ks", "2s", "3c")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.FOUR_OF_A_KIND

    def test_straight_flush(self):
        """Test straight flush detection."""
        hole = (Card.from_string("9h").value, Card.from_string("8h").value)
        board = self._cards("Th", "Jh", "Qh", "2s", "3c")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.STRAIGHT_FLUSH

    def test_wheel_straight(self):
        """Test wheel (A-2-3-4-5) straight."""
        hole = (Card.from_string("Ah").value, Card.from_string("2d").value)
        board = self._cards("3c", "4h", "5s", "Ts", "Jc")
        strength = self.evaluator.evaluate(hole, board)
        rank = strength // (10**10)
        assert rank == HandRank.STRAIGHT


class TestPokerGame:
    """Test main game engine."""

    def setup_method(self):
        """Set up game."""
        self.game = PokerGame()
        self.game.seed(42)

    def test_reset_hand(self):
        """Test hand reset."""
        state = self.game.reset_hand()

        assert state is not None
        assert state.street == Street.PREFLOP
        assert len(state.players) == 6
        assert state.pot == 1.5  # SB + BB
        assert state.current_bet == 1.0

        # All players should have cards
        for player in state.players:
            assert player.hole_cards is not None
            assert len(player.hole_cards) == 2

    def test_legal_actions_preflop(self):
        """Test legal actions preflop."""
        self.game.reset_hand()
        legal = self.game.get_legal_actions()

        # UTG can fold, raise, or allin (not check/call since no one ahead)
        assert legal[ActionType.FOLD] is True
        assert legal[ActionType.RAISE] is True
        assert legal[ActionType.ALLIN] is True
        # Postflop actions not available preflop
        assert legal[ActionType.RAISE_33] is False
        assert legal[ActionType.RAISE_75] is False

    def test_fold_action(self):
        """Test fold action."""
        state = self.game.reset_hand()
        current = state.current_player

        state, rewards, done = self.game.step(ActionType.FOLD)

        assert not state.players[current].is_active
        assert not done  # Game continues

    def test_raise_action(self):
        """Test raise action."""
        state = self.game.reset_hand()
        initial_pot = state.pot

        state, rewards, done = self.game.step(ActionType.RAISE)

        assert state.pot > initial_pot
        assert state.raise_count == 1

    def test_complete_hand(self):
        """Test playing a complete hand."""
        self.game.reset_hand()

        # Everyone folds to BB
        for _ in range(5):
            state = self.game.get_state()
            if state.hand_over:
                break

            legal = self.game.get_legal_actions()
            if legal[ActionType.FOLD]:
                state, rewards, done = self.game.step(ActionType.FOLD)
            elif legal[ActionType.CHECK_CALL]:
                state, rewards, done = self.game.step(ActionType.CHECK_CALL)

            if done:
                break

        # Hand should be over
        assert self.game.get_state().hand_over

    def test_showdown(self):
        """Test showdown scenario."""
        self.game.reset_hand()

        done = False
        max_actions = 100

        for _ in range(max_actions):
            if done:
                break

            legal = self.game.get_legal_actions()

            # Choose check/call to get to showdown
            if legal[ActionType.CHECK_CALL]:
                _, _, done = self.game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                _, _, done = self.game.step(ActionType.FOLD)

        # Hand should complete
        assert done or self.game.get_state().hand_over

    def test_player_positions(self):
        """Test player position calculation."""
        state = self.game.reset_hand()

        # Button should be at correct position
        btn_pos = state.get_player_position(state.button_seat)
        assert btn_pos == Position.BTN

        # SB is next to button
        sb_seat = (state.button_seat + 1) % 6
        sb_pos = state.get_player_position(sb_seat)
        assert sb_pos == Position.SB

        # BB is two after button
        bb_seat = (state.button_seat + 2) % 6
        bb_pos = state.get_player_position(bb_seat)
        assert bb_pos == Position.BB

    def test_allin_action(self):
        """Test all-in action."""
        state = self.game.reset_hand()
        current = state.current_player
        initial_stack = state.players[current].stack

        state, rewards, done = self.game.step(ActionType.ALLIN)

        assert state.players[current].stack == 0
        assert state.players[current].is_allin

    def test_state_for_player(self):
        """Test getting state from player perspective."""
        self.game.reset_hand()
        state_dict = self.game.get_state_for_player(0)

        assert "hole_cards" in state_dict
        assert "board" in state_dict
        assert "my_position" in state_dict
        assert "pot" in state_dict
        assert "opponents" in state_dict
        assert len(state_dict["opponents"]) == 5


class TestPotCalculation:
    """Test pot calculation scenarios."""

    def test_simple_pot(self):
        """Test simple pot calculation."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # Initial pot should be 1.5 BB (0.5 SB + 1.0 BB)
        assert state.pot == 1.5

    def test_raised_pot(self):
        """Test pot after raise."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # UTG raises (2.5 BB)
        state, _, _ = game.step(ActionType.RAISE)

        # Pot should be 1.5 + 2.5 = 4.0
        assert state.pot == 4.0


class TestPotFunctions:
    """Test pot management functions directly."""

    def _make_player(self, seat, stack=100, bet_this_street=0, bet_total=0,
                     is_active=True, is_allin=False):
        """Helper to create PlayerState with all required fields."""
        from sixmax.engine.state import PlayerState
        return PlayerState(
            seat=seat,
            stack=stack,
            bet_this_street=bet_this_street,
            bet_total=bet_total,
            is_active=is_active,
            is_allin=is_allin,
            hole_cards=None,
        )

    def test_calculate_side_pots_simple(self):
        """Test side pot calculation with simple all-in."""
        from sixmax.engine.pot import calculate_side_pots

        players = [
            self._make_player(0, stack=0, bet_total=50, is_active=True, is_allin=True),
            self._make_player(1, stack=50, bet_total=100, is_active=True, is_allin=False),
            self._make_player(2, stack=0, bet_total=100, is_active=True, is_allin=True),
        ]

        side_pots = calculate_side_pots(players)

        # Main pot: 50 * 3 = 150 (all 3 eligible)
        # Side pot: 50 * 2 = 100 (only seats 1, 2 eligible)
        assert len(side_pots) >= 1
        total = sum(pot for pot, _ in side_pots)
        assert total == 250

    def test_calculate_side_pots_empty(self):
        """Test side pot with no active players."""
        from sixmax.engine.pot import calculate_side_pots

        players = [
            self._make_player(0, stack=100, bet_total=0, is_active=False),
            self._make_player(1, stack=100, bet_total=0, is_active=False),
        ]

        side_pots = calculate_side_pots(players)
        assert side_pots == []

    def test_calculate_side_pots_no_bets(self):
        """Test side pot with active players but no bets."""
        from sixmax.engine.pot import calculate_side_pots

        players = [
            self._make_player(0, stack=100, bet_total=0, is_active=True),
            self._make_player(1, stack=100, bet_total=0, is_active=True),
        ]

        side_pots = calculate_side_pots(players)
        assert side_pots == []

    def test_calculate_total_pot(self):
        """Test total pot calculation."""
        from sixmax.engine.pot import calculate_total_pot

        players = [
            self._make_player(0, stack=50, bet_total=50),
            self._make_player(1, stack=0, bet_total=100),
            self._make_player(2, stack=75, bet_total=25),
        ]

        total = calculate_total_pot(players)
        assert total == 175

    def test_distribute_pot_single_winner(self):
        """Test pot distribution to single winner."""
        from sixmax.engine.pot import distribute_pot

        players = [self._make_player(i) for i in range(3)]
        winnings = distribute_pot(100.0, [1], players)

        assert winnings == {1: 100.0}

    def test_distribute_pot_multiple_winners(self):
        """Test pot distribution to multiple winners (split pot)."""
        from sixmax.engine.pot import distribute_pot

        players = [self._make_player(i) for i in range(3)]
        winnings = distribute_pot(100.0, [0, 2], players)

        assert winnings == {0: 50.0, 2: 50.0}

    def test_distribute_pot_no_winners(self):
        """Test pot distribution with no winners."""
        from sixmax.engine.pot import distribute_pot

        players = [self._make_player(i) for i in range(3)]
        winnings = distribute_pot(100.0, [], players)

        assert winnings == {}

    def test_distribute_all_pots(self):
        """Test distributing multiple side pots."""
        from sixmax.engine.pot import distribute_all_pots

        side_pots = [
            (150.0, [0, 1, 2]),  # Main pot
            (100.0, [1, 2]),     # Side pot
        ]

        # Seat 0 has best hand for main pot, seat 2 wins side pot
        hand_rankings = {
            0: 1000,  # Best hand but only eligible for main pot
            1: 500,
            2: 800,
        }

        winnings = distribute_all_pots(side_pots, hand_rankings)

        # Seat 0 wins main pot (150)
        # Seat 2 wins side pot (100)
        assert winnings[0] == 150.0
        assert winnings[2] == 100.0
        assert 1 not in winnings or winnings.get(1, 0) == 0

    def test_distribute_all_pots_tie(self):
        """Test distributing pots with tied hands."""
        from sixmax.engine.pot import distribute_all_pots

        side_pots = [
            (100.0, [0, 1]),
        ]

        # Both have same hand strength
        hand_rankings = {
            0: 1000,
            1: 1000,
        }

        winnings = distribute_all_pots(side_pots, hand_rankings)

        assert winnings[0] == 50.0
        assert winnings[1] == 50.0

    def test_distribute_all_pots_no_eligible(self):
        """Test pot distribution when no one has hand rankings."""
        from sixmax.engine.pot import distribute_all_pots

        side_pots = [
            (100.0, [0, 1]),
        ]

        # No rankings provided (e.g., all folded before showdown)
        hand_rankings = {}

        winnings = distribute_all_pots(side_pots, hand_rankings)

        assert winnings == {}


class TestPostflopActions:
    """Test postflop action handling."""

    def setup_method(self):
        """Set up game."""
        self.game = PokerGame()
        self.game.seed(42)

    def _get_to_flop(self) -> bool:
        """Play until we reach the flop. Returns True if reached flop."""
        self.game.reset_hand()
        raised_once = False
        for _ in range(20):
            state = self.game.get_state()
            if state.hand_over:
                return False
            if state.street != Street.PREFLOP:
                return True
            legal = self.game.get_legal_actions()
            # Need at least one raise preflop (no limp allowed)
            if not raised_once and legal[ActionType.RAISE]:
                self.game.step(ActionType.RAISE)
                raised_once = True
            elif legal[ActionType.CHECK_CALL]:
                self.game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                self.game.step(ActionType.FOLD)
        return self.game.get_state().street != Street.PREFLOP

    def test_postflop_check(self):
        """Test checking on the flop."""
        if not self._get_to_flop():
            pytest.skip("Could not reach flop")

        state = self.game.get_state()
        legal = self.game.get_legal_actions()

        # Should be able to check on flop with no bet
        if state.current_bet == 0 or (state.current_bet - state.players[state.current_player].bet_this_street) == 0:
            assert legal[ActionType.CHECK_CALL]
            old_pot = state.pot
            state, _, done = self.game.step(ActionType.CHECK_CALL)
            # Pot should remain the same after check
            assert state.pot == old_pot

    def test_postflop_bet_33(self):
        """Test 33% pot bet on flop."""
        if not self._get_to_flop():
            pytest.skip("Could not reach flop")

        state = self.game.get_state()
        legal = self.game.get_legal_actions()

        # 33% pot bet should be available
        if legal[ActionType.RAISE_33]:
            old_pot = state.pot
            state, _, done = self.game.step(ActionType.RAISE_33)
            assert state.pot > old_pot
            assert state.raise_count >= 1

    def test_postflop_bet_75(self):
        """Test 75% pot bet on flop."""
        if not self._get_to_flop():
            pytest.skip("Could not reach flop")

        state = self.game.get_state()
        legal = self.game.get_legal_actions()

        # 75% pot bet should be available
        if legal[ActionType.RAISE_75]:
            old_pot = state.pot
            state, _, done = self.game.step(ActionType.RAISE_75)
            assert state.pot > old_pot

    def test_postflop_call_after_bet(self):
        """Test calling a bet on the flop."""
        if not self._get_to_flop():
            pytest.skip("Could not reach flop")

        # First player bets
        legal = self.game.get_legal_actions()
        if legal[ActionType.RAISE_33]:
            self.game.step(ActionType.RAISE_33)

            # Next player should be able to call
            state = self.game.get_state()
            if not state.hand_over:
                legal = self.game.get_legal_actions()
                if legal[ActionType.CHECK_CALL]:
                    old_pot = state.pot
                    state, _, _ = self.game.step(ActionType.CHECK_CALL)
                    assert state.pot > old_pot

    def test_postflop_raise_after_bet(self):
        """Test raising after a bet on flop."""
        if not self._get_to_flop():
            pytest.skip("Could not reach flop")

        # First player bets
        legal = self.game.get_legal_actions()
        if legal[ActionType.RAISE_33]:
            self.game.step(ActionType.RAISE_33)

            # Next player should be able to raise
            state = self.game.get_state()
            if not state.hand_over:
                legal = self.game.get_legal_actions()
                # Try to raise (33 or 75)
                if legal[ActionType.RAISE_75]:
                    old_pot = state.pot
                    old_raise_count = state.raise_count
                    state, _, _ = self.game.step(ActionType.RAISE_75)
                    assert state.pot > old_pot
                    assert state.raise_count > old_raise_count


class TestPreflopEdgeCases:
    """Test preflop edge cases."""

    def setup_method(self):
        """Set up game."""
        self.game = PokerGame()
        self.game.seed(42)

    def test_bb_check_option(self):
        """Test BB can check when limped around."""
        self.game.reset_hand()

        # All players call until BB
        for _ in range(5):
            state = self.game.get_state()
            if state.hand_over or state.street != Street.PREFLOP:
                break

            bb_seat = (state.button_seat + 2) % 6
            current = state.current_player

            # If we're at BB position
            if current == bb_seat:
                legal = self.game.get_legal_actions()
                # BB should have check option if limped around
                to_call = state.current_bet - state.players[current].bet_this_street
                if to_call == 0:
                    assert legal[ActionType.CHECK_CALL]
                break

            legal = self.game.get_legal_actions()
            if legal[ActionType.CHECK_CALL]:
                self.game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                self.game.step(ActionType.FOLD)

    def test_multiple_reraises(self):
        """Test multiple re-raises (3bet, 4bet)."""
        self.game.reset_hand()

        # Count successful raises
        raise_count = 0
        for _ in range(10):
            state = self.game.get_state()
            if state.hand_over or state.street != Street.PREFLOP:
                break

            legal = self.game.get_legal_actions()
            if legal[ActionType.RAISE]:
                self.game.step(ActionType.RAISE)
                raise_count += 1
            elif legal[ActionType.CHECK_CALL]:
                self.game.step(ActionType.CHECK_CALL)
            else:
                break

        # Should have at least one raise
        assert raise_count >= 1

    def test_short_stack_allin(self):
        """Test all-in action works correctly."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Find a player that can all-in
        allin_tested = False
        for _ in range(10):
            state = game.get_state()
            if state.hand_over:
                break

            legal = game.get_legal_actions()
            current_player_seat = state.current_player

            if legal[ActionType.ALLIN] and not allin_tested:
                old_state = game.get_state()
                old_stack = old_state.players[current_player_seat].stack
                game.step(ActionType.ALLIN)
                new_state = game.get_state()
                # Verify all-in worked - the player who went all-in should be marked
                assert new_state.players[current_player_seat].is_allin
                assert new_state.players[current_player_seat].stack == 0
                allin_tested = True
                break
            elif legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)
            else:
                break

        assert allin_tested, "Should have tested all-in action"


class TestHandComparison:
    """Test hand comparison and showdown logic."""

    def setup_method(self):
        """Set up evaluator."""
        self.evaluator = HandEvaluator()

    def _cards(self, *names: str) -> list[int]:
        """Convert card names to integers."""
        return [Card.from_string(n).value for n in names]

    def test_compare_two_hands(self):
        """Test comparing two hands."""
        from sixmax.engine import compare_hands

        board = self._cards("2c", "5h", "8s", "Ts", "Jc")
        hands = [
            ((Card.from_string("Ah").value, Card.from_string("Ad").value), 0),  # Pair of Aces
            ((Card.from_string("Kh").value, Card.from_string("Kd").value), 1),  # Pair of Kings
        ]

        winners = compare_hands(hands, board)
        assert winners == [0]  # Aces win

    def test_compare_tie(self):
        """Test tie in hand comparison."""
        from sixmax.engine import compare_hands

        # Board makes the best hand
        board = self._cards("As", "Ks", "Qs", "Js", "Ts")  # Royal flush on board
        hands = [
            ((Card.from_string("2h").value, Card.from_string("3h").value), 0),
            ((Card.from_string("4h").value, Card.from_string("5h").value), 1),
        ]

        winners = compare_hands(hands, board)
        # Both should tie with royal flush on board
        assert len(winners) == 2
        assert 0 in winners
        assert 1 in winners

    def test_hand_name(self):
        """Test getting hand name."""
        hole = (Card.from_string("Ah").value, Card.from_string("Ad").value)
        board = self._cards("Ac", "Kh", "Ks", "2s", "3c")
        strength = self.evaluator.evaluate(hole, board)

        name = self.evaluator.get_hand_name(strength)
        assert name == "Full House"


class TestGameState:
    """Test GameState methods."""

    def test_count_active_players(self):
        """Test counting active players."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # Initially all 6 players active
        assert state.count_active_players() == 6

        # After one fold
        game.step(ActionType.FOLD)
        state = game.get_state()
        assert state.count_active_players() == 5

    def test_get_active_players(self):
        """Test getting active players list."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # All seats should be active initially
        active = state.get_active_players()
        assert len(active) == 6

        # After fold, should have 5
        game.step(ActionType.FOLD)
        state = game.get_state()
        active = state.get_active_players()
        assert len(active) == 5


class TestAllInScenarios:
    """Test all-in scenarios and side pots."""

    def test_heads_up_allin(self):
        """Test heads up all-in situation."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Fold everyone except two players, then all-in
        for _ in range(20):
            state = game.get_state()
            if state.hand_over:
                break

            active_count = state.count_active_players()
            legal = game.get_legal_actions()

            if active_count <= 2 and legal[ActionType.ALLIN]:
                game.step(ActionType.ALLIN)
            elif legal[ActionType.FOLD] and active_count > 2:
                game.step(ActionType.FOLD)
            elif legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            else:
                break

        # Game should complete
        assert game.get_state().hand_over or game.get_state().count_active_players() <= 2


class TestCardUtilities:
    """Test card utility functions."""

    def test_card_repr(self):
        """Test card string representation."""
        card = Card(0)  # 2 of spades
        assert str(card) == "2s"

        card = Card(51)  # Ace of clubs
        assert str(card) == "Ac"

    def test_card_from_invalid_string(self):
        """Test card from invalid string."""
        with pytest.raises((KeyError, IndexError, ValueError)):
            Card.from_string("XX")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
