"""Core data structures for the poker game engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Street(IntEnum):
    """Game street (betting round)."""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class Position(IntEnum):
    """6-max position relative to button."""

    UTG = 0  # Under the Gun
    HJ = 1  # Hijack
    CO = 2  # Cutoff
    BTN = 3  # Button
    SB = 4  # Small Blind
    BB = 5  # Big Blind


class ActionType(IntEnum):
    """
    Unified 6-dimensional action encoding.

    Preflop: FOLD, CHECK_CALL, RAISE, ALLIN (R33/R75 not available)
    Postflop: FOLD, CHECK_CALL, R33, R75, ALLIN (RAISE not available)
    """

    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2  # Preflop only
    RAISE_33 = 3  # Postflop only: 33% pot
    RAISE_75 = 4  # Postflop only: 75% pot
    ALLIN = 5


NUM_ACTIONS = 6
NUM_PLAYERS = 6
STARTING_STACK = 100.0  # BB units


@dataclass(slots=True)
class Card:
    """
    Card representation.

    Internal: integer 0-51
    - card = rank * 4 + suit
    - rank: 0-12 (2,3,4,5,6,7,8,9,T,J,Q,K,A)
    - suit: 0-3 (spade, heart, diamond, club)
    """

    value: int

    @property
    def rank(self) -> int:
        """Get rank (0-12)."""
        return self.value // 4

    @property
    def suit(self) -> int:
        """Get suit (0-3)."""
        return self.value % 4

    @classmethod
    def from_string(cls, s: str) -> Card:
        """Create card from string like 'As', 'Kh', '2d'."""
        ranks = "23456789TJQKA"
        suits = "shdc"
        rank = ranks.index(s[0].upper())
        suit = suits.index(s[1].lower())
        return cls(rank * 4 + suit)

    def __repr__(self) -> str:
        ranks = "23456789TJQKA"
        suits = "shdc"
        return f"{ranks[self.rank]}{suits[self.suit]}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(self.value)


# Padding values for encoding
RANK_PADDING = 13
SUIT_PADDING = 4
CARD_PADDING = 52


@dataclass(slots=True)
class PlayerState:
    """Single player state."""

    seat: int  # Seat number 0-5
    stack: float  # Current chips (BB units)
    bet_this_street: float  # Bet amount this street
    bet_total: float  # Total invested this hand
    is_active: bool  # Still in hand (not folded)
    is_allin: bool  # Already all-in
    hole_cards: tuple[int, int] | None  # Hand cards (2 cards)

    def reset_street(self) -> None:
        """Reset street-specific state."""
        self.bet_this_street = 0.0

    def reset_hand(self, stack: float) -> None:
        """Reset for new hand."""
        self.stack = stack
        self.bet_this_street = 0.0
        self.bet_total = 0.0
        self.is_active = True
        self.is_allin = False
        self.hole_cards = None


@dataclass
class ActionRecord:
    """Record of a single action."""

    player: int  # Player seat
    action: ActionType  # Action type
    amount: float  # Actual amount (BB units)
    street: Street  # Which street


@dataclass
class GameState:
    """Complete game state."""

    # Player info
    players: list[PlayerState]

    # Current state
    street: Street
    current_player: int  # Current acting player seat
    button_seat: int  # Button position seat

    # Pot info
    pot: float  # Main pot
    side_pots: list[tuple[float, list[int]]]  # [(pot_size, [eligible_players])]
    current_bet: float  # Current street max bet
    min_raise: float  # Minimum raise amount

    # Board
    board: list[int] = field(default_factory=list)  # Community cards (0-5)

    # Action history
    actions: list[ActionRecord] = field(default_factory=list)
    raise_count: int = 0  # Raise count this street (for preflop sizing)

    # Internal
    deck: list[int] = field(default_factory=list)
    hand_over: bool = False

    def get_active_players(self) -> list[int]:
        """Get list of active (not folded) player seats."""
        return [p.seat for p in self.players if p.is_active]

    def get_active_not_allin_players(self) -> list[int]:
        """Get list of active players who are not all-in."""
        return [p.seat for p in self.players if p.is_active and not p.is_allin]

    def count_active_players(self) -> int:
        """Count active players."""
        return sum(1 for p in self.players if p.is_active)

    def get_player_position(self, seat: int) -> Position:
        """Get position for a given seat."""
        offset = (seat - self.button_seat) % NUM_PLAYERS
        # BTN=0, SB=1, BB=2, UTG=3, HJ=4, CO=5
        pos_map = {0: Position.BTN, 1: Position.SB, 2: Position.BB, 3: Position.UTG, 4: Position.HJ, 5: Position.CO}
        return pos_map[offset]
