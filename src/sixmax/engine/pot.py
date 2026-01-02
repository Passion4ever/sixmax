"""Pot management for Texas Hold'em."""

from __future__ import annotations

from .state import PlayerState


def calculate_side_pots(
    players: list[PlayerState],
) -> list[tuple[float, list[int]]]:
    """
    Calculate side pots when players are all-in.

    Args:
        players: List of all players

    Returns:
        List of (pot_size, eligible_seats) tuples, ordered from main pot to side pots
    """
    # Get all unique bet amounts from active players
    active_players = [p for p in players if p.is_active or p.bet_total > 0]

    if not active_players:
        return []

    # Collect all distinct investment levels
    bet_levels = sorted(set(p.bet_total for p in active_players if p.bet_total > 0))

    if not bet_levels:
        return []

    side_pots: list[tuple[float, list[int]]] = []
    prev_level = 0.0

    for level in bet_levels:
        # Calculate contribution at this level
        layer_size = level - prev_level

        # Count contributors at this level
        contributors = [p for p in active_players if p.bet_total >= level]
        pot_contribution = layer_size * len(contributors)

        # Eligible players are active players who have bet at least this level
        eligible_seats = [p.seat for p in players if p.is_active and p.bet_total >= level]

        if pot_contribution > 0 and eligible_seats:
            side_pots.append((pot_contribution, eligible_seats))

        prev_level = level

    return side_pots


def calculate_total_pot(players: list[PlayerState]) -> float:
    """Calculate total pot from all player contributions."""
    return sum(p.bet_total for p in players)


def distribute_pot(
    pot_amount: float, winner_seats: list[int], players: list[PlayerState]
) -> dict[int, float]:
    """
    Distribute pot to winners.

    Args:
        pot_amount: Amount to distribute
        winner_seats: List of winning seat numbers
        players: All players

    Returns:
        Dict of seat -> winnings
    """
    if not winner_seats:
        return {}

    share = pot_amount / len(winner_seats)
    return {seat: share for seat in winner_seats}


def distribute_all_pots(
    side_pots: list[tuple[float, list[int]]],
    hand_rankings: dict[int, int],  # seat -> hand strength
) -> dict[int, float]:
    """
    Distribute all pots based on hand rankings.

    Args:
        side_pots: List of (pot_size, eligible_seats) from calculate_side_pots
        hand_rankings: Dict of seat -> hand strength value

    Returns:
        Dict of seat -> total winnings
    """
    winnings: dict[int, float] = {}

    for pot_size, eligible_seats in side_pots:
        # Filter to only eligible seats that have rankings
        eligible_with_hands = [s for s in eligible_seats if s in hand_rankings]

        if not eligible_with_hands:
            continue

        # Find best hand among eligible
        best_strength = max(hand_rankings[s] for s in eligible_with_hands)
        winners = [s for s in eligible_with_hands if hand_rankings[s] == best_strength]

        # Distribute this pot
        share = pot_size / len(winners)
        for seat in winners:
            winnings[seat] = winnings.get(seat, 0.0) + share

    return winnings
