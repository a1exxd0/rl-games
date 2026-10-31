import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace

from tienlen.engine.cards import DECK, HAND_SIZE, TWO_RANK, Card, format_cards, rank_of
from tienlen.engine.combos import PASS, InvalidPlay, Play, beats, enumerate_plays, is_chop
from tienlen.engine.rules import Rules
from tienlen.engine.state import GameState, Move, PlayerView

INSTANT_WIN_PAIRS = 6


class IllegalPlay(ValueError):
    pass


def _ring(seat: int, n_players: int) -> list[int]:
    return [(seat + offset) % n_players for offset in range(1, n_players + 1)]


def instant_win(hand: Iterable[Card]) -> str | None:
    counts = Counter(rank_of(card) for card in hand)
    if counts[TWO_RANK] == 4:
        return "four_twos"
    if sum(count // 2 for count in counts.values()) >= INSTANT_WIN_PAIRS:
        return "six_pairs"
    if all(counts[rank] for rank in range(TWO_RANK)):
        return "dragon"
    return None


def new_game(rules: Rules, seed: int) -> GameState:
    deck = list(DECK)
    random.Random(seed).shuffle(deck)
    n_players = rules.n_players
    hands = tuple(
        tuple(sorted(deck[seat * HAND_SIZE : (seat + 1) * HAND_SIZE])) for seat in range(n_players)
    )
    leader = min(range(n_players), key=lambda seat: hands[seat][0])
    state = GameState(
        rules=rules,
        seed=seed,
        hands=hands,
        undealt=tuple(sorted(deck[n_players * HAND_SIZE :])),
        first_leader=leader,
        to_act=leader,
    )
    if rules.instant_wins:
        order = _ring(leader - 1, n_players)
        winner = next((seat for seat in order if instant_win(hands[seat])), None)
        if winner is not None:
            return replace(state, to_act=None, placements=tuple(_ring(winner - 1, n_players)))
    return state


def _is_locked(state: GameState, seat: int) -> bool:
    return state.rules.lock_on_pass and seat in state.passed


def _fits_table(state: GameState, hand: tuple[Card, ...], play: Play) -> bool:
    if state.current is None:
        must_show_lowest = state.rules.first_lead_includes_lowest and not state.moves
        return not must_show_lowest or hand[0] in play.cards
    if _is_locked(state, state.to_act):
        return is_chop(play, state.current)
    return beats(play, state.current)


def _finish_allowed(state: GameState, hand: tuple[Card, ...], play: Play) -> bool:
    if not state.rules.forbid_finish_on_two or len(play.cards) != len(hand):
        return True
    forced = state.current is None and len(hand) == 1
    return forced or rank_of(play.top) != TWO_RANK


def _allowed(state: GameState, hand: tuple[Card, ...], play: Play) -> bool:
    return _fits_table(state, hand, play) and _finish_allowed(state, hand, play)


def legal_plays(state: GameState) -> list[Play]:
    if state.to_act is None:
        return []
    hand = state.hands[state.to_act]
    plays = [play for play in enumerate_plays(hand) if _allowed(state, hand, play)]
    if state.current is not None:
        plays.append(PASS)
    return plays


def is_legal(state: GameState, play: Play) -> bool:
    if state.to_act is None:
        return False
    if play.is_pass:
        return state.current is not None
    hand = state.hands[state.to_act]
    try:
        canonical = Play.of(play.cards)
    except InvalidPlay:
        return False
    return canonical == play and set(play.cards).issubset(hand) and _allowed(state, hand, play)


def _may_respond(state: GameState, seat: int) -> bool:
    if not state.hands[seat]:
        return False
    if not _is_locked(state, seat):
        return True
    return state.rules.chop_after_pass and any(
        is_chop(play, state.current) for play in enumerate_plays(state.hands[seat])
    )


def _next_turn(state: GameState) -> GameState:
    for seat in _ring(state.to_act, state.rules.n_players):
        if seat == state.last_player:
            order = _ring(seat - 1, state.rules.n_players)
            return replace(
                state,
                to_act=next(s for s in order if state.hands[s]),
                current=None,
                last_player=None,
                passed=frozenset(),
                round_start=len(state.moves),
            )
        if _may_respond(state, seat):
            return replace(state, to_act=seat)
    raise AssertionError("the ring always reaches the last player")


def apply(state: GameState, play: Play) -> GameState:
    if not is_legal(state, play):
        raise IllegalPlay(f"seat {state.to_act} cannot play {format_cards(play.cards) or 'pass'}")
    seat = state.to_act
    moves = (*state.moves, Move(seat, play))
    if play.is_pass:
        return _next_turn(replace(state, passed=state.passed | {seat}, moves=moves))

    hand = tuple(card for card in state.hands[seat] if card not in play.cards)
    hands = (*state.hands[:seat], hand, *state.hands[seat + 1 :])
    placements = state.placements if hand else (*state.placements, seat)
    passed = state.passed - {seat} if state.rules.lock_on_pass else frozenset()
    after = replace(
        state,
        hands=hands,
        current=play,
        last_player=seat,
        passed=passed,
        placements=placements,
        moves=moves,
    )
    holders = [s for s, cards in enumerate(hands) if cards]
    if len(holders) == 1:
        return replace(after, to_act=None, placements=(*placements, holders[0]))
    return _next_turn(after)


def view(state: GameState, seat: int) -> PlayerView:
    return PlayerView(
        rules=state.rules,
        seat=seat,
        hand=state.hands[seat],
        hand_counts=tuple(len(hand) for hand in state.hands),
        first_leader=state.first_leader,
        to_act=state.to_act,
        current=state.current,
        last_player=state.last_player,
        passed=state.passed,
        placements=state.placements,
        moves=state.moves,
        round_start=state.round_start,
    )


def is_over(state: GameState) -> bool:
    return state.to_act is None


def placements(state: GameState) -> tuple[int, ...]:
    return state.placements


def replay(rules: Rules, seed: int, moves: Iterable[Move]) -> GameState:
    state = new_game(rules, seed)
    for move in moves:
        if move.seat != state.to_act:
            raise IllegalPlay(f"seat {move.seat} moved out of turn; expected {state.to_act}")
        state = apply(state, move.play)
    return state
