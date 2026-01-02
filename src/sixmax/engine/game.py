"""Main poker game engine."""

from __future__ import annotations

import random
from typing import Any

from .actions import (
    apply_action,
    get_action_order,
    get_legal_actions,
    get_next_player,
    is_street_complete,
)
from .evaluator import HandEvaluator
from .pot import calculate_side_pots, distribute_all_pots
from .state import (
    NUM_ACTIONS,
    NUM_PLAYERS,
    STARTING_STACK,
    ActionType,
    GameState,
    PlayerState,
    Position,
    Street,
)


class PokerGame:
    """
    Texas Hold'em Poker Game Engine.

    Implements 6-max no-limit hold'em with 100BB starting stacks.

    Main API:
    - reset_hand(): Start a new hand
    - step(action): Execute an action
    - get_state(): Get current game state
    - get_legal_actions(): Get legal action mask
    """

    def __init__(
        self,
        num_players: int = NUM_PLAYERS,
        starting_stack: float = STARTING_STACK,
        button_seat: int = 0,
    ):
        """
        Initialize poker game.

        Args:
            num_players: Number of players (default 6)
            starting_stack: Starting stack in BB (default 100)
            button_seat: Initial button position (default 0)
        """
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.button_seat = button_seat
        self.evaluator = HandEvaluator()

        # Initialize players
        self.players = [
            PlayerState(
                seat=i,
                stack=starting_stack,
                bet_this_street=0.0,
                bet_total=0.0,
                is_active=True,
                is_allin=False,
                hole_cards=None,
            )
            for i in range(num_players)
        ]

        self.state: GameState | None = None
        self._rng = random.Random()

    def seed(self, seed: int | None = None) -> None:
        """Set random seed for reproducibility."""
        self._rng.seed(seed)

    def reset_hand(self) -> GameState:
        """
        Start a new hand.

        Returns:
            Initial game state
        """
        # Move button
        self.button_seat = (self.button_seat + 1) % self.num_players

        # Reset players
        for player in self.players:
            player.reset_hand(self.starting_stack)

        # Post blinds
        sb_seat = (self.button_seat + 1) % self.num_players
        bb_seat = (self.button_seat + 2) % self.num_players
        self._post_blind(sb_seat, 0.5)
        self._post_blind(bb_seat, 1.0)

        # Shuffle and deal
        deck = list(range(52))
        self._rng.shuffle(deck)

        for player in self.players:
            player.hole_cards = (deck.pop(), deck.pop())

        # First to act preflop is UTG
        first_to_act = (self.button_seat + 3) % self.num_players

        # Create game state
        self.state = GameState(
            players=self.players,
            street=Street.PREFLOP,
            current_player=first_to_act,
            button_seat=self.button_seat,
            pot=1.5,  # SB + BB
            side_pots=[],
            current_bet=1.0,  # BB
            min_raise=1.0,  # Min raise is 1 BB
            board=[],
            actions=[],
            raise_count=0,
            deck=deck,
            hand_over=False,
        )

        return self.state

    def step(self, action: ActionType | int) -> tuple[GameState, dict[int, float], bool]:
        """
        Execute an action.

        Args:
            action: Action to take (ActionType or int)

        Returns:
            Tuple of (new_state, rewards, done)
            - rewards: Dict of seat -> reward (only non-zero at hand end)
            - done: Whether hand is complete
        """
        if self.state is None:
            raise RuntimeError("Game not started. Call reset_hand() first.")

        if self.state.hand_over:
            raise RuntimeError("Hand is over. Call reset_hand() to start new hand.")

        if isinstance(action, int):
            action = ActionType(action)

        # Validate action
        legal = get_legal_actions(self.state)
        if not legal[action]:
            raise ValueError(
                f"Illegal action {action.name} for player {self.state.current_player}"
            )

        # Apply action
        apply_action(self.state, action)

        # Check if hand ends (only one player left)
        if self.state.count_active_players() <= 1:
            return self._end_hand_single_winner()

        # Check if street is complete
        if is_street_complete(self.state):
            if self.state.street == Street.RIVER:
                # Showdown
                return self._showdown()
            else:
                # Move to next street
                self._next_street()
        else:
            # Find next player to act
            next_player = get_next_player(self.state)
            if next_player is not None:
                self.state.current_player = next_player
            else:
                # Street complete, advance
                if self.state.street == Street.RIVER:
                    return self._showdown()
                else:
                    self._next_street()

        return self.state, {}, False

    def get_state(self) -> GameState:
        """Get current game state."""
        if self.state is None:
            raise RuntimeError("Game not started. Call reset_hand() first.")
        return self.state

    def get_legal_actions(self) -> list[bool]:
        """Get legal action mask for current player."""
        if self.state is None:
            return [False] * NUM_ACTIONS
        return get_legal_actions(self.state)

    def get_current_player(self) -> int:
        """Get current player seat."""
        if self.state is None:
            raise RuntimeError("Game not started.")
        return self.state.current_player

    def get_state_for_player(self, player_id: int) -> dict[str, Any]:
        """
        Get player-perspective state (hides other players' hole cards).

        Used for neural network input.
        """
        if self.state is None:
            raise RuntimeError("Game not started.")

        player = self.players[player_id]

        return {
            "hole_cards": player.hole_cards,
            "board": self.state.board.copy(),
            "my_position": self.state.get_player_position(player_id),
            "my_stack": player.stack,
            "pot": self.state.pot,
            "to_call": max(0, self.state.current_bet - player.bet_this_street),
            "street": self.state.street,
            "raise_count": self.state.raise_count,
            "current_bet": self.state.current_bet,
            "is_my_turn": self.state.current_player == player_id,
            "opponents": [
                {
                    "seat": p.seat,
                    "position": self.state.get_player_position(p.seat),
                    "is_active": p.is_active,
                    "stack": p.stack,
                    "invested": p.bet_total,
                    "bet_this_street": p.bet_this_street,
                    "is_allin": p.is_allin,
                }
                for p in self.players
                if p.seat != player_id
            ],
            "actions": self.state.actions.copy(),
        }

    def _post_blind(self, seat: int, amount: float) -> None:
        """Post a blind bet."""
        player = self.players[seat]
        actual_amount = min(amount, player.stack)
        player.stack -= actual_amount
        player.bet_this_street = actual_amount
        player.bet_total = actual_amount

        if player.stack == 0:
            player.is_allin = True

    def _next_street(self) -> None:
        """Advance to the next street."""
        if self.state is None:
            return

        # Reset street state
        for player in self.players:
            player.reset_street()

        self.state.current_bet = 0.0
        self.state.min_raise = 1.0  # Reset to 1 BB
        self.state.raise_count = 0

        # Deal community cards
        if self.state.street == Street.PREFLOP:
            # Flop: deal 3 cards
            self.state.street = Street.FLOP
            self.state.board.extend(
                [self.state.deck.pop() for _ in range(3)]
            )
        elif self.state.street == Street.FLOP:
            # Turn: deal 1 card
            self.state.street = Street.TURN
            self.state.board.append(self.state.deck.pop())
        elif self.state.street == Street.TURN:
            # River: deal 1 card
            self.state.street = Street.RIVER
            self.state.board.append(self.state.deck.pop())

        # Find first player to act (first active player after button)
        active_seats = self.state.get_active_not_allin_players()
        if active_seats:
            order = get_action_order(
                self.state.street, self.state.button_seat, active_seats
            )
            if order:
                self.state.current_player = order[0]

    def _end_hand_single_winner(self) -> tuple[GameState, dict[int, float], bool]:
        """End hand when only one player remains."""
        if self.state is None:
            return self.state, {}, True  # type: ignore

        self.state.hand_over = True

        # Find the winner
        active = self.state.get_active_players()
        if not active:
            return self.state, {}, True

        winner = active[0]

        # Calculate rewards
        rewards = {}
        for player in self.players:
            if player.seat == winner:
                # Winner gets pot minus their investment
                rewards[player.seat] = self.state.pot - player.bet_total
            else:
                # Losers lose their investment
                rewards[player.seat] = -player.bet_total

        return self.state, rewards, True

    def _showdown(self) -> tuple[GameState, dict[int, float], bool]:
        """Handle showdown at river."""
        if self.state is None:
            return self.state, {}, True  # type: ignore

        self.state.hand_over = True

        # Get active players' hands
        active_players = [p for p in self.players if p.is_active]

        if len(active_players) <= 1:
            return self._end_hand_single_winner()

        # Evaluate hands
        hand_rankings = {}
        for player in active_players:
            if player.hole_cards is not None:
                strength = self.evaluator.evaluate(player.hole_cards, self.state.board)
                hand_rankings[player.seat] = strength

        # Calculate side pots
        side_pots = calculate_side_pots(self.players)

        # If no side pots calculated, use main pot
        if not side_pots:
            side_pots = [(self.state.pot, [p.seat for p in active_players])]

        # Distribute pots
        winnings = distribute_all_pots(side_pots, hand_rankings)

        # Calculate rewards (profit/loss)
        rewards = {}
        for player in self.players:
            win_amount = winnings.get(player.seat, 0.0)
            rewards[player.seat] = win_amount - player.bet_total

        return self.state, rewards, True
