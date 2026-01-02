"""Action processing logic for the poker game engine."""

from __future__ import annotations

from .state import (
    NUM_ACTIONS,
    NUM_PLAYERS,
    ActionRecord,
    ActionType,
    GameState,
    Street,
)

# Preflop raise sizes (BB units)
PREFLOP_RAISE_SIZES = [2.5, 9.0, 22.0, 100.0]  # Open, 3bet, 4bet, 5bet+


def get_preflop_raise_size(raise_count: int) -> float:
    """
    Get preflop raise size based on raise count.

    Args:
        raise_count: Number of raises so far this street

    Returns:
        Raise-to amount in BB
    """
    return PREFLOP_RAISE_SIZES[min(raise_count, len(PREFLOP_RAISE_SIZES) - 1)]


def calculate_postflop_bet_size(state: GameState, ratio: float) -> float:
    """
    Calculate postflop bet/raise size.

    Args:
        state: Current game state
        ratio: Bet ratio (0.33 or 0.75)

    Returns:
        Bet amount in BB

    Note:
        - Bet (no facing bet): ratio × pot
        - Raise (facing bet): R33 uses 50% pot, R75 uses 75% pot
    """
    player = state.players[state.current_player]
    to_call = state.current_bet - player.bet_this_street

    if to_call == 0:
        # Bet: ratio × pot
        return ratio * state.pot
    else:
        # Raise: R33 -> 50% pot, R75 -> 75% pot (unchanged)
        effective_ratio = 0.50 if ratio == 0.33 else ratio
        return effective_ratio * state.pot


def get_legal_actions(state: GameState) -> list[bool]:
    """
    Get legal action mask for current player.

    Returns:
        6-dimensional boolean mask [FOLD, CHECK_CALL, RAISE, R33, R75, ALLIN]
    """
    mask = [False] * NUM_ACTIONS
    player = state.players[state.current_player]
    to_call = state.current_bet - player.bet_this_street

    # Player is all-in or not active - no actions
    if player.is_allin or not player.is_active:
        return mask

    # Not enough chips to call - can only fold or all-in
    if player.stack <= to_call:
        mask[ActionType.FOLD] = True
        mask[ActionType.ALLIN] = True
        return mask

    if state.street == Street.PREFLOP:
        # === Preflop rules ===
        bb_seat = (state.button_seat + 2) % NUM_PLAYERS

        if state.raise_count > 0:
            # Facing a raise (someone has raised beyond BB)
            mask[ActionType.FOLD] = True
            mask[ActionType.CHECK_CALL] = True  # Call

            # Can re-raise if raise count < 3 (allows up to 4bet, then only all-in)
            if state.raise_count < 3:
                # Check if player has enough for min raise
                raise_to = get_preflop_raise_size(state.raise_count)
                if player.stack > raise_to - player.bet_this_street:
                    mask[ActionType.RAISE] = True

            mask[ActionType.ALLIN] = True
        else:
            # No one has raised yet (only blinds posted)
            if player.seat == bb_seat and to_call == 0:
                # BB can check if no one raised
                mask[ActionType.CHECK_CALL] = True  # Check
                mask[ActionType.RAISE] = True
                mask[ActionType.ALLIN] = True
            else:
                # First to act (open) - NO LIMP allowed per DESIGN.md
                mask[ActionType.FOLD] = True
                mask[ActionType.RAISE] = True  # Open raise
                mask[ActionType.ALLIN] = True
    else:
        # === Postflop rules ===
        if to_call > 0:
            # Facing a bet
            mask[ActionType.FOLD] = True
            mask[ActionType.CHECK_CALL] = True  # Call

            # Can raise if have enough chips
            bet_33 = calculate_postflop_bet_size(state, 0.33)
            bet_75 = calculate_postflop_bet_size(state, 0.75)
            min_raise_to = state.current_bet + state.min_raise

            if player.stack > to_call + bet_33:
                mask[ActionType.RAISE_33] = True
            if player.stack > to_call + bet_75:
                mask[ActionType.RAISE_75] = True

            mask[ActionType.ALLIN] = True
        else:
            # No bet yet
            mask[ActionType.CHECK_CALL] = True  # Check
            mask[ActionType.RAISE_33] = True  # Bet 33%
            mask[ActionType.RAISE_75] = True  # Bet 75%
            mask[ActionType.ALLIN] = True

    return mask


def apply_action(state: GameState, action: ActionType) -> float:
    """
    Apply action to game state (mutates state).

    Args:
        state: Current game state (will be modified)
        action: Action to apply

    Returns:
        Actual amount bet/raised
    """
    player = state.players[state.current_player]
    to_call = state.current_bet - player.bet_this_street
    actual_amount = 0.0

    if action == ActionType.FOLD:
        player.is_active = False
        actual_amount = 0.0

    elif action == ActionType.CHECK_CALL:
        call_amount = min(to_call, player.stack)
        player.stack -= call_amount
        player.bet_this_street += call_amount
        player.bet_total += call_amount
        state.pot += call_amount
        actual_amount = call_amount

        if player.stack == 0:
            player.is_allin = True

    elif action == ActionType.RAISE:
        # Preflop raise
        raise_to = get_preflop_raise_size(state.raise_count)
        bet_amount = raise_to - player.bet_this_street

        # Cap at player's stack
        if bet_amount >= player.stack:
            bet_amount = player.stack
            player.is_allin = True
            raise_to = player.bet_this_street + bet_amount

        player.stack -= bet_amount
        player.bet_this_street = raise_to
        player.bet_total += bet_amount
        state.pot += bet_amount
        state.current_bet = raise_to
        state.min_raise = raise_to - (state.current_bet - state.min_raise)  # Update min raise
        state.raise_count += 1
        actual_amount = bet_amount

    elif action == ActionType.RAISE_33:
        # Postflop 33% pot bet/raise
        bet_size = calculate_postflop_bet_size(state, 0.33)
        actual_amount = _execute_postflop_bet(state, player, bet_size, to_call)

    elif action == ActionType.RAISE_75:
        # Postflop 75% pot bet/raise
        bet_size = calculate_postflop_bet_size(state, 0.75)
        actual_amount = _execute_postflop_bet(state, player, bet_size, to_call)

    elif action == ActionType.ALLIN:
        allin_amount = player.stack
        old_bet = player.bet_this_street

        player.bet_this_street += allin_amount
        player.bet_total += allin_amount
        player.stack = 0
        player.is_allin = True
        state.pot += allin_amount

        if player.bet_this_street > state.current_bet:
            # This is a raise
            raise_size = player.bet_this_street - state.current_bet
            state.min_raise = max(state.min_raise, raise_size)
            state.current_bet = player.bet_this_street
            state.raise_count += 1

        actual_amount = allin_amount

    # Record action
    state.actions.append(
        ActionRecord(
            player=state.current_player,
            action=action,
            amount=actual_amount,
            street=state.street,
        )
    )

    return actual_amount


def _execute_postflop_bet(state: GameState, player, bet_size: float, to_call: float) -> float:
    """Execute a postflop bet/raise."""
    if to_call == 0:
        # This is a bet
        total_bet = bet_size
    else:
        # This is a raise
        total_bet = to_call + bet_size

    # Cap at player's stack
    if total_bet >= player.stack:
        total_bet = player.stack
        player.is_allin = True

    new_bet_this_street = player.bet_this_street + total_bet
    player.stack -= total_bet
    player.bet_this_street = new_bet_this_street
    player.bet_total += total_bet
    state.pot += total_bet

    if new_bet_this_street > state.current_bet:
        raise_size = new_bet_this_street - state.current_bet
        state.min_raise = max(state.min_raise, raise_size)
        state.current_bet = new_bet_this_street
        state.raise_count += 1

    return total_bet


def get_action_order(street: Street, button: int, active_seats: list[int]) -> list[int]:
    """
    Get action order for a street.

    Args:
        street: Current street
        button: Button seat
        active_seats: List of active player seats

    Returns:
        Ordered list of seats to act
    """
    if street == Street.PREFLOP:
        # Preflop: UTG → HJ → CO → BTN → SB → BB
        order = [
            (button + 3) % NUM_PLAYERS,  # UTG
            (button + 4) % NUM_PLAYERS,  # HJ
            (button + 5) % NUM_PLAYERS,  # CO
            button,  # BTN
            (button + 1) % NUM_PLAYERS,  # SB
            (button + 2) % NUM_PLAYERS,  # BB
        ]
    else:
        # Postflop: SB → BB → UTG → HJ → CO → BTN
        order = [
            (button + 1) % NUM_PLAYERS,  # SB
            (button + 2) % NUM_PLAYERS,  # BB
            (button + 3) % NUM_PLAYERS,  # UTG
            (button + 4) % NUM_PLAYERS,  # HJ
            (button + 5) % NUM_PLAYERS,  # CO
            button,  # BTN
        ]

    return [seat for seat in order if seat in active_seats]


def get_next_player(state: GameState) -> int | None:
    """
    Get next player to act.

    Returns:
        Next player seat, or None if street/hand is complete
    """
    active_seats = state.get_active_not_allin_players()

    if len(active_seats) == 0:
        return None

    if len(active_seats) == 1:
        # Only one player left who can act
        player = state.players[active_seats[0]]
        if player.bet_this_street >= state.current_bet:
            return None
        return active_seats[0]

    # Get action order
    order = get_action_order(state.street, state.button_seat, active_seats)

    # Find current player in order
    try:
        current_idx = order.index(state.current_player)
    except ValueError:
        current_idx = -1

    # Look for next player who needs to act
    for i in range(1, len(order) + 1):
        next_idx = (current_idx + i) % len(order)
        next_seat = order[next_idx]
        next_player = state.players[next_seat]

        # Player needs to act if they haven't matched current bet
        # or if they haven't acted this round yet
        if next_player.bet_this_street < state.current_bet:
            return next_seat

    # Check if we've gone full circle (everyone has acted and matched)
    return None


def is_street_complete(state: GameState) -> bool:
    """Check if current street betting is complete."""
    active_not_allin = state.get_active_not_allin_players()

    # If no one left to act, street is complete
    if len(active_not_allin) == 0:
        return True

    # If only one player not all-in and they've matched, complete
    if len(active_not_allin) == 1:
        player = state.players[active_not_allin[0]]
        return player.bet_this_street >= state.current_bet

    # Check if all active non-allin players have matched the bet
    for seat in active_not_allin:
        player = state.players[seat]
        if player.bet_this_street < state.current_bet:
            return False

    # Need to check if everyone has had a chance to act
    # This is tracked by whether the next player to act would be None
    return get_next_player(state) is None
