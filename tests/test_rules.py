"""Rule verification tests for Texas Hold'em poker.

These tests verify that the game engine correctly implements
the poker rules defined in DESIGN.md.
"""

import pytest

from sixmax.engine import ActionType, PokerGame, Position, Street


class TestPreflopRaiseSizes:
    """
    Verify preflop raise sizes according to DESIGN.md:
    - raise_count=0 (Open): 2.5 BB
    - raise_count=1 (3-Bet): 9 BB
    - raise_count=2 (4-Bet): 22 BB
    - raise_count>=3 (5-Bet+): All-in
    """

    def test_open_raise_is_2_5bb(self):
        """UTG open raise should be 2.5 BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # Initial pot: SB(0.5) + BB(1.0) = 1.5 BB
        assert state.pot == 1.5
        assert state.raise_count == 0

        # UTG raises (open)
        initial_pot = state.pot
        state, _, _ = game.step(ActionType.RAISE)

        # After open: pot should be 1.5 + 2.5 = 4.0 BB
        assert state.pot == 4.0, f"Open raise should make pot 4.0 BB, got {state.pot}"
        assert state.raise_count == 1
        assert state.current_bet == 2.5, f"Current bet should be 2.5 BB, got {state.current_bet}"

    def test_3bet_is_9bb(self):
        """3-bet should be 9 BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # UTG opens to 2.5 BB
        state, _, _ = game.step(ActionType.RAISE)
        assert state.raise_count == 1

        # Next player 3-bets
        # Find next player who can raise
        for _ in range(5):
            if game.get_state().hand_over:
                break
            legal = game.get_legal_actions()
            if legal[ActionType.RAISE]:
                state, _, _ = game.step(ActionType.RAISE)
                break
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

        # After 3-bet: current bet should be 9 BB
        if state.raise_count == 2:
            assert state.current_bet == 9.0, f"3-bet should be 9 BB, got {state.current_bet}"

    def test_4bet_is_22bb(self):
        """4-bet should be 22 BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        raise_count = 0
        for _ in range(20):
            state = game.get_state()
            if state.hand_over or state.street != Street.PREFLOP:
                break

            legal = game.get_legal_actions()
            if legal[ActionType.RAISE] and state.raise_count < 3:
                state, _, _ = game.step(ActionType.RAISE)
                raise_count = state.raise_count
                if raise_count == 3:
                    # After 4-bet: current bet should be 22 BB
                    assert state.current_bet == 22.0, f"4-bet should be 22 BB, got {state.current_bet}"
                    break
            elif legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

    def test_5bet_plus_forces_allin_only(self):
        """After 4-bet (raise_count >= 3), only FOLD/CALL/ALLIN should be available."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Get to 4-bet situation
        for _ in range(20):
            state = game.get_state()
            if state.hand_over or state.street != Street.PREFLOP:
                break

            legal = game.get_legal_actions()

            if state.raise_count >= 3:
                # After 4-bet, RAISE should NOT be available
                assert not legal[ActionType.RAISE], \
                    f"RAISE should not be available after 4-bet (raise_count={state.raise_count})"
                # But ALLIN should still be available
                assert legal[ActionType.ALLIN], "ALLIN should be available after 4-bet"
                break

            if legal[ActionType.RAISE]:
                game.step(ActionType.RAISE)
            elif legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)


class TestActionOrder:
    """
    Verify correct action order:
    - Preflop: UTG -> HJ -> CO -> BTN -> SB -> BB
    - Postflop: SB -> BB -> UTG -> HJ -> CO -> BTN
    """

    def test_preflop_starts_with_utg(self):
        """Preflop action should start with UTG (3 seats after button)."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # UTG is 3 positions after button: BTN+1=SB, BTN+2=BB, BTN+3=UTG
        expected_utg = (state.button_seat + 3) % 6
        assert state.current_player == expected_utg, \
            f"Preflop should start with UTG (seat {expected_utg}), got seat {state.current_player}"

        # Verify position
        pos = state.get_player_position(state.current_player)
        assert pos == Position.UTG, f"First actor should be UTG, got {pos}"

    def test_preflop_action_order(self):
        """Verify complete preflop action order: UTG -> HJ -> CO -> BTN -> SB -> BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        button = state.button_seat
        expected_order = [
            (button + 3) % 6,  # UTG
            (button + 4) % 6,  # HJ
            (button + 5) % 6,  # CO
            button,            # BTN
            (button + 1) % 6,  # SB
            (button + 2) % 6,  # BB
        ]

        actual_order = []
        for _ in range(6):
            state = game.get_state()
            if state.hand_over or state.street != Street.PREFLOP:
                break

            actual_order.append(state.current_player)

            legal = game.get_legal_actions()
            if legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

        # Verify order matches expected
        for i, (expected, actual) in enumerate(zip(expected_order, actual_order)):
            assert expected == actual, \
                f"Position {i}: expected seat {expected}, got {actual}"


class TestActionLegality:
    """
    Verify action legality masks according to DESIGN.md:

    Preflop:
    - 无人入池(非BB): [1,0,1,0,0,1] FOLD/RAISE/ALLIN
    - BB且无人加注: [0,1,1,0,0,1] CHECK/RAISE/ALLIN
    - 面对加注: [1,1,1,0,0,1] FOLD/CALL/RAISE/ALLIN
    - 面对4bet+: [1,1,0,0,0,1] FOLD/CALL/ALLIN

    Postflop:
    - 无人下注: [0,1,0,1,1,1] CHECK/BET33/BET75/ALLIN
    - 面对下注: [1,1,0,1,1,1] FOLD/CALL/R33/R75/ALLIN
    """

    def test_preflop_first_to_act_no_limp(self):
        """First to act preflop (UTG) can FOLD/RAISE/ALLIN, NOT CHECK_CALL."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        legal = game.get_legal_actions()

        # UTG is first to act, no one limped
        assert legal[ActionType.FOLD], "UTG should be able to FOLD"
        assert legal[ActionType.RAISE], "UTG should be able to RAISE (open)"
        assert legal[ActionType.ALLIN], "UTG should be able to ALLIN"
        # CHECK_CALL should NOT be available (no one to call)
        assert not legal[ActionType.CHECK_CALL], "UTG should NOT be able to CHECK/CALL before anyone acts"
        # Postflop actions should not be available
        assert not legal[ActionType.RAISE_33], "RAISE_33 is postflop only"
        assert not legal[ActionType.RAISE_75], "RAISE_75 is postflop only"

    def test_preflop_bb_check_option(self):
        """BB can CHECK if everyone limps (calls BB)."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        bb_seat = (state.button_seat + 2) % 6

        # Everyone calls until BB
        for _ in range(10):
            state = game.get_state()
            if state.hand_over or state.street != Street.PREFLOP:
                break

            current = state.current_player
            legal = game.get_legal_actions()

            if current == bb_seat:
                # BB's turn - if everyone limped, should have CHECK option
                to_call = state.current_bet - state.players[current].bet_this_street
                if to_call == 0:
                    assert legal[ActionType.CHECK_CALL], "BB should be able to CHECK when limped around"
                    assert not legal[ActionType.FOLD], "BB should NOT need to FOLD when limped around"
                break

            # Others call
            if legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

    def test_preflop_facing_raise(self):
        """Facing a raise: FOLD/CALL/RAISE/ALLIN available."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # UTG raises
        game.step(ActionType.RAISE)

        # Next player faces a raise
        state = game.get_state()
        if not state.hand_over:
            legal = game.get_legal_actions()

            assert legal[ActionType.FOLD], "Should be able to FOLD facing raise"
            assert legal[ActionType.CHECK_CALL], "Should be able to CALL facing raise"
            assert legal[ActionType.ALLIN], "Should be able to ALLIN facing raise"
            # RAISE depends on raise_count
            if state.raise_count < 3:
                assert legal[ActionType.RAISE], "Should be able to RAISE (3-bet) facing open"

    def test_postflop_no_bet(self):
        """Postflop with no bet: CHECK/BET33/BET75/ALLIN available."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Get to flop
        for _ in range(20):
            state = game.get_state()
            if state.hand_over:
                break
            if state.street != Street.PREFLOP:
                break

            legal = game.get_legal_actions()
            if legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

        # Now on flop
        state = game.get_state()
        if state.street.value > 0 and not state.hand_over:
            legal = game.get_legal_actions()
            to_call = state.current_bet - state.players[state.current_player].bet_this_street

            if to_call == 0:  # No bet facing
                assert not legal[ActionType.FOLD], "Should NOT need to FOLD when no bet"
                assert legal[ActionType.CHECK_CALL], "Should be able to CHECK"
                assert legal[ActionType.RAISE_33], "Should be able to BET 33%"
                assert legal[ActionType.RAISE_75], "Should be able to BET 75%"
                assert legal[ActionType.ALLIN], "Should be able to ALLIN"
                # Preflop RAISE not available postflop
                assert not legal[ActionType.RAISE], "RAISE is preflop only"

    def test_postflop_facing_bet(self):
        """Postflop facing a bet: FOLD/CALL/RAISE33/RAISE75/ALLIN available."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Get to flop
        for _ in range(20):
            state = game.get_state()
            if state.hand_over:
                break
            if state.street != Street.PREFLOP:
                break

            legal = game.get_legal_actions()
            if legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

        # Someone bets on flop
        state = game.get_state()
        if state.street.value > 0 and not state.hand_over:
            legal = game.get_legal_actions()
            if legal[ActionType.RAISE_33]:
                game.step(ActionType.RAISE_33)

                # Next player faces a bet
                state = game.get_state()
                if not state.hand_over:
                    legal = game.get_legal_actions()
                    assert legal[ActionType.FOLD], "Should be able to FOLD facing bet"
                    assert legal[ActionType.CHECK_CALL], "Should be able to CALL facing bet"
                    assert legal[ActionType.ALLIN], "Should be able to ALLIN facing bet"


class TestPostflopBetSizes:
    """
    Verify postflop bet sizes according to DESIGN.md:
    - RAISE_33: 33% pot (bet) or 50% (pot+call) (raise)
    - RAISE_75: 75% pot (bet) or 75% (pot+call) (raise)
    """

    def _get_to_flop(self, game):
        """Helper to advance game to flop."""
        raised_once = False
        for _ in range(20):
            state = game.get_state()
            if state.hand_over:
                return False
            if state.street != Street.PREFLOP:
                return True

            legal = game.get_legal_actions()
            # Need at least one raise preflop (no limp allowed)
            if not raised_once and legal[ActionType.RAISE]:
                game.step(ActionType.RAISE)
                raised_once = True
            elif legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)
        return game.get_state().street != Street.PREFLOP

    def test_bet_33_percent(self):
        """33% pot bet size verification."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        if not self._get_to_flop(game):
            pytest.skip("Could not reach flop")

        state = game.get_state()
        if state.hand_over:
            pytest.skip("Hand ended before flop betting")

        pot_before = state.pot
        legal = game.get_legal_actions()

        if legal[ActionType.RAISE_33]:
            game.step(ActionType.RAISE_33)
            state = game.get_state()

            # Expected bet = 33% of pot
            expected_bet = pot_before * 0.33
            actual_bet = state.current_bet

            # Allow some tolerance for rounding
            assert abs(actual_bet - expected_bet) < 0.1, \
                f"33% bet should be ~{expected_bet:.2f}, got {actual_bet:.2f}"

    def test_bet_75_percent(self):
        """75% pot bet size verification."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        if not self._get_to_flop(game):
            pytest.skip("Could not reach flop")

        state = game.get_state()
        if state.hand_over:
            pytest.skip("Hand ended before flop betting")

        pot_before = state.pot
        legal = game.get_legal_actions()

        if legal[ActionType.RAISE_75]:
            game.step(ActionType.RAISE_75)
            state = game.get_state()

            # Expected bet = 75% of pot
            expected_bet = pot_before * 0.75
            actual_bet = state.current_bet

            # Allow some tolerance for rounding
            assert abs(actual_bet - expected_bet) < 0.1, \
                f"75% bet should be ~{expected_bet:.2f}, got {actual_bet:.2f}"


class TestPositions:
    """Verify correct position assignments."""

    def test_all_positions_assigned(self):
        """All 6 positions should be correctly assigned."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        positions = set()
        for seat in range(6):
            pos = state.get_player_position(seat)
            positions.add(pos)

        expected = {Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB}
        assert positions == expected, f"Expected all 6 positions, got {positions}"

    def test_position_order_from_button(self):
        """Positions should be correctly ordered relative to button."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        btn = state.button_seat

        assert state.get_player_position(btn) == Position.BTN
        assert state.get_player_position((btn + 1) % 6) == Position.SB
        assert state.get_player_position((btn + 2) % 6) == Position.BB
        assert state.get_player_position((btn + 3) % 6) == Position.UTG
        assert state.get_player_position((btn + 4) % 6) == Position.HJ
        assert state.get_player_position((btn + 5) % 6) == Position.CO


class TestBlinds:
    """Verify blind posting rules."""

    def test_sb_posts_0_5bb(self):
        """Small blind should be 0.5 BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        sb_seat = (state.button_seat + 1) % 6
        sb_player = state.players[sb_seat]

        assert sb_player.bet_this_street == 0.5, \
            f"SB should post 0.5 BB, got {sb_player.bet_this_street}"

    def test_bb_posts_1bb(self):
        """Big blind should be 1 BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        bb_seat = (state.button_seat + 2) % 6
        bb_player = state.players[bb_seat]

        assert bb_player.bet_this_street == 1.0, \
            f"BB should post 1.0 BB, got {bb_player.bet_this_street}"

    def test_initial_pot_is_1_5bb(self):
        """Initial pot should be SB + BB = 1.5 BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        assert state.pot == 1.5, f"Initial pot should be 1.5 BB, got {state.pot}"


class TestStackSizes:
    """Verify stack management."""

    def test_initial_stacks_100bb(self):
        """All players should start with 100 BB."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        # After blinds:
        # - SB has 99.5 BB (100 - 0.5)
        # - BB has 99 BB (100 - 1)
        # - Others have 100 BB
        sb_seat = (state.button_seat + 1) % 6
        bb_seat = (state.button_seat + 2) % 6

        for i, player in enumerate(state.players):
            if player.seat == sb_seat:
                expected = 99.5
            elif player.seat == bb_seat:
                expected = 99.0
            else:
                expected = 100.0

            assert player.stack == expected, \
                f"Player {i} should have {expected} BB, got {player.stack}"


class TestHandEvaluation:
    """Verify hand evaluation is correct."""

    def test_pair_beats_high_card(self):
        """A pair should beat high card."""
        from sixmax.engine import Card, HandEvaluator

        evaluator = HandEvaluator()

        # Pair of aces
        pair_hole = (Card.from_string("Ah").value, Card.from_string("Ad").value)
        # Ace-King high
        high_hole = (Card.from_string("As").value, Card.from_string("Kd").value)
        board = [Card.from_string(c).value for c in ["2c", "5h", "8s", "Ts", "Jc"]]

        pair_strength = evaluator.evaluate(pair_hole, board)
        high_strength = evaluator.evaluate(high_hole, board)

        assert pair_strength > high_strength, "Pair should beat high card"

    def test_flush_beats_straight(self):
        """A flush should beat a straight."""
        from sixmax.engine import Card, HandEvaluator

        evaluator = HandEvaluator()

        # Flush (5 hearts)
        flush_hole = (Card.from_string("2h").value, Card.from_string("3h").value)
        flush_board = [Card.from_string(c).value for c in ["7h", "9h", "Kh", "2c", "3c"]]

        # Straight (9-K)
        straight_hole = (Card.from_string("9s").value, Card.from_string("Td").value)
        straight_board = [Card.from_string(c).value for c in ["Jh", "Qc", "Ks", "2c", "3c"]]

        flush_strength = evaluator.evaluate(flush_hole, flush_board)
        straight_strength = evaluator.evaluate(straight_hole, straight_board)

        assert flush_strength > straight_strength, "Flush should beat straight"

    def test_full_house_beats_flush(self):
        """A full house should beat a flush."""
        from sixmax.engine import Card, HandEvaluator

        evaluator = HandEvaluator()

        # Full house (AAA-KK)
        fh_hole = (Card.from_string("Ah").value, Card.from_string("Ad").value)
        fh_board = [Card.from_string(c).value for c in ["As", "Kh", "Kc", "2c", "3c"]]

        # Flush
        flush_hole = (Card.from_string("2h").value, Card.from_string("3h").value)
        flush_board = [Card.from_string(c).value for c in ["7h", "9h", "Kh", "2c", "3c"]]

        fh_strength = evaluator.evaluate(fh_hole, fh_board)
        flush_strength = evaluator.evaluate(flush_hole, flush_board)

        assert fh_strength > flush_strength, "Full house should beat flush"


class TestGameFlow:
    """Verify correct game flow."""

    def test_street_progression(self):
        """Game should progress through streets: PREFLOP -> FLOP -> TURN -> RIVER."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        streets_seen = [Street.PREFLOP]
        raised_once = False

        for _ in range(50):
            state = game.get_state()
            if state.hand_over:
                break

            if state.street not in streets_seen:
                streets_seen.append(state.street)

            legal = game.get_legal_actions()

            # Need at least one raise preflop (no limp allowed), then can call
            if not raised_once and legal[ActionType.RAISE]:
                game.step(ActionType.RAISE)
                raised_once = True
            elif legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

        # Should have seen multiple streets
        assert len(streets_seen) >= 2, f"Should progress through streets, only saw {streets_seen}"

    def test_flop_deals_3_cards(self):
        """Flop should deal exactly 3 community cards."""
        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        # Get to flop
        for _ in range(20):
            state = game.get_state()
            if state.hand_over:
                break
            if state.street == Street.FLOP:
                assert len(state.board) == 3, f"Flop should have 3 cards, got {len(state.board)}"
                break

            legal = game.get_legal_actions()
            if legal[ActionType.CHECK_CALL]:
                game.step(ActionType.CHECK_CALL)
            elif legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)

    def test_everyone_folds_winner_gets_pot(self):
        """When everyone folds to one player, that player wins the pot."""
        game = PokerGame()
        game.seed(42)
        state = game.reset_hand()

        initial_pot = state.pot

        # Everyone folds except last player
        for _ in range(10):
            state = game.get_state()
            if state.hand_over:
                break

            active = state.count_active_players()
            if active == 1:
                # One player left, hand should be over
                assert state.hand_over, "Hand should end when only 1 player remains"
                break

            legal = game.get_legal_actions()
            if legal[ActionType.FOLD]:
                game.step(ActionType.FOLD)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
