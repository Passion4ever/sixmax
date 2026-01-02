"""Hand evaluator for Texas Hold'em."""

from __future__ import annotations

from enum import IntEnum
from itertools import combinations


class HandRank(IntEnum):
    """Hand ranking (higher is better)."""

    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


# Hand rank multiplier for comparison
RANK_MULTIPLIER = 10**10


class HandEvaluator:
    """
    Hand evaluator using direct calculation.

    For production, consider using lookup tables or
    Two Plus Two evaluator for better performance.
    """

    def evaluate(self, hole_cards: tuple[int, int], board: list[int]) -> int:
        """
        Evaluate the best 5-card hand.

        Args:
            hole_cards: Player's hole cards (2 cards as integers 0-51)
            board: Community cards (3-5 cards as integers 0-51)

        Returns:
            Hand strength value (higher is better)
        """
        all_cards = list(hole_cards) + list(board)

        if len(all_cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(all_cards)}")

        best_rank = 0

        # Try all 5-card combinations
        for combo in combinations(all_cards, 5):
            rank = self._evaluate_5_cards(combo)
            if rank > best_rank:
                best_rank = rank

        return best_rank

    def _evaluate_5_cards(self, cards: tuple[int, ...]) -> int:
        """
        Evaluate exactly 5 cards.

        Returns a value where higher is better.
        Format: hand_rank * RANK_MULTIPLIER + kicker_value
        """
        # Extract ranks and suits
        ranks = sorted([c // 4 for c in cards], reverse=True)
        suits = [c % 4 for c in cards]

        # Check for flush
        is_flush = len(set(suits)) == 1

        # Check for straight
        is_straight, straight_high = self._check_straight(ranks)

        # Count rank occurrences
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        counts = sorted(rank_counts.values(), reverse=True)
        # Get ranks by count (for kicker comparison)
        ranks_by_count = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)

        # Determine hand rank
        if is_straight and is_flush:
            return HandRank.STRAIGHT_FLUSH * RANK_MULTIPLIER + straight_high

        if counts == [4, 1]:
            # Four of a kind
            quad_rank = ranks_by_count[0]
            kicker = ranks_by_count[1]
            return HandRank.FOUR_OF_A_KIND * RANK_MULTIPLIER + quad_rank * 13 + kicker

        if counts == [3, 2]:
            # Full house
            trips_rank = ranks_by_count[0]
            pair_rank = ranks_by_count[1]
            return HandRank.FULL_HOUSE * RANK_MULTIPLIER + trips_rank * 13 + pair_rank

        if is_flush:
            return HandRank.FLUSH * RANK_MULTIPLIER + self._kicker_value(ranks)

        if is_straight:
            return HandRank.STRAIGHT * RANK_MULTIPLIER + straight_high

        if counts == [3, 1, 1]:
            # Three of a kind
            trips_rank = ranks_by_count[0]
            kickers = sorted([r for r in ranks if r != trips_rank], reverse=True)
            return (
                HandRank.THREE_OF_A_KIND * RANK_MULTIPLIER
                + trips_rank * (13**2)
                + kickers[0] * 13
                + kickers[1]
            )

        if counts == [2, 2, 1]:
            # Two pair
            pairs = sorted([r for r in ranks_by_count if rank_counts[r] == 2], reverse=True)
            kicker = [r for r in ranks_by_count if rank_counts[r] == 1][0]
            return (
                HandRank.TWO_PAIR * RANK_MULTIPLIER + pairs[0] * (13**2) + pairs[1] * 13 + kicker
            )

        if counts == [2, 1, 1, 1]:
            # One pair
            pair_rank = ranks_by_count[0]
            kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
            return (
                HandRank.ONE_PAIR * RANK_MULTIPLIER
                + pair_rank * (13**3)
                + kickers[0] * (13**2)
                + kickers[1] * 13
                + kickers[2]
            )

        # High card
        return HandRank.HIGH_CARD * RANK_MULTIPLIER + self._kicker_value(ranks)

    def _check_straight(self, ranks: list[int]) -> tuple[bool, int]:
        """
        Check for straight.

        Args:
            ranks: Sorted ranks (high to low)

        Returns:
            (is_straight, high_card)
        """
        unique_ranks = sorted(set(ranks), reverse=True)

        if len(unique_ranks) < 5:
            return False, 0

        # Check regular straight
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i + 4] == 4:
                return True, unique_ranks[i]

        # Check wheel (A-2-3-4-5)
        if set(unique_ranks) >= {12, 0, 1, 2, 3}:  # A, 2, 3, 4, 5
            return True, 3  # 5 is high card (rank 3)

        return False, 0

    def _kicker_value(self, ranks: list[int]) -> int:
        """Calculate kicker value for comparison."""
        value = 0
        for i, r in enumerate(sorted(ranks, reverse=True)[:5]):
            value += r * (13 ** (4 - i))
        return value

    def compare_hands(
        self, hands: list[tuple[tuple[int, int], int]], board: list[int]
    ) -> list[int]:
        """
        Compare multiple hands and return winners.

        Args:
            hands: List of (hole_cards, seat) tuples
            board: Community cards

        Returns:
            List of winning seat numbers (can be multiple for ties)
        """
        evaluations = []
        for hole_cards, seat in hands:
            strength = self.evaluate(hole_cards, board)
            evaluations.append((strength, seat))

        # Find max strength
        max_strength = max(e[0] for e in evaluations)

        # Return all seats with max strength
        return [seat for strength, seat in evaluations if strength == max_strength]

    def get_hand_name(self, strength: int) -> str:
        """Get human-readable hand name from strength value."""
        hand_rank = HandRank(strength // RANK_MULTIPLIER)
        names = {
            HandRank.HIGH_CARD: "High Card",
            HandRank.ONE_PAIR: "One Pair",
            HandRank.TWO_PAIR: "Two Pair",
            HandRank.THREE_OF_A_KIND: "Three of a Kind",
            HandRank.STRAIGHT: "Straight",
            HandRank.FLUSH: "Flush",
            HandRank.FULL_HOUSE: "Full House",
            HandRank.FOUR_OF_A_KIND: "Four of a Kind",
            HandRank.STRAIGHT_FLUSH: "Straight Flush",
        }
        return names.get(hand_rank, "Unknown")


# Singleton evaluator instance
_evaluator = HandEvaluator()


def evaluate_hand(hole_cards: tuple[int, int], board: list[int]) -> int:
    """Convenience function to evaluate a hand."""
    return _evaluator.evaluate(hole_cards, board)


def compare_hands(hands: list[tuple[tuple[int, int], int]], board: list[int]) -> list[int]:
    """Convenience function to compare hands."""
    return _evaluator.compare_hands(hands, board)
