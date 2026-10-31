import hashlib
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

from tienlen.bots.interface import TienLenBot
from tienlen.engine import (
    PASS,
    GameState,
    Move,
    Play,
    Rules,
    apply,
    is_over,
    legal_plays,
    new_game,
    placements,
    view,
)

FORFEIT = object()


class MoveTimeout(BaseException):
    pass


@dataclass(frozen=True)
class MatchResult:
    seed: int
    rules_id: str
    seats: tuple[str, ...]
    placements: tuple[int, ...]
    moves: tuple[Move, ...]
    forfeits: tuple[int, ...]
    rules: Rules | None = None


def play_match(
    bots: Sequence[TienLenBot],
    rules: Rules,
    seed: int,
    move_timeout: float | None = None,
    seats: Sequence[str] | None = None,
) -> MatchResult:
    if len(bots) != rules.n_players:
        raise ValueError(f"expected {rules.n_players} bots, got {len(bots)}")
    forfeits = _reset_all(bots, rules, seed, move_timeout)
    state = new_game(rules, seed)
    while not is_over(state):
        state = apply(state, _choose(bots[state.actor], state, forfeits, move_timeout))
    return MatchResult(
        seed=seed,
        rules_id=rules.rules_id,
        seats=tuple(seats or (type(bot).__name__ for bot in bots)),
        placements=_final_order(state, forfeits),
        moves=state.moves,
        forfeits=tuple(forfeits),
        rules=rules,
    )


def bot_seed(deal_seed: int, seat: int) -> int:
    digest = hashlib.sha256(f"{deal_seed}:{seat}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _reset_all(
    bots: Sequence[TienLenBot], rules: Rules, seed: int, timeout: float | None
) -> list[int]:
    return [
        seat
        for seat, bot in enumerate(bots)
        if _attempt(partial(bot.reset, seat, rules, bot_seed(seed, seat)), timeout) is FORFEIT
    ]


def _choose(bot: TienLenBot, state: GameState, forfeits: list[int], timeout: float | None) -> Play:
    seat = state.actor
    legal = legal_plays(state)
    if seat in forfeits:
        return _fallback(legal)
    seen = view(state, seat)
    choice = _attempt(lambda: bot.act(seen, tuple(legal)), timeout)
    if choice is not FORFEIT and choice in legal:
        return choice
    forfeits.append(seat)
    return _fallback(legal)


def _fallback(legal: Sequence[Play]) -> Play:
    if PASS in legal:
        return PASS
    return min(legal, key=lambda play: (play.top, len(play.cards)))


def _final_order(state: GameState, forfeits: Sequence[int]) -> tuple[int, ...]:
    finishers = (seat for seat in placements(state) if seat not in forfeits)
    return (*finishers, *reversed(forfeits))


def _attempt(call: Callable[[], Any], timeout: float | None) -> Any:
    started = time.monotonic()
    try:
        with _deadline(timeout):
            outcome = call()
    except (Exception, MoveTimeout):  # noqa: BLE001
        return FORFEIT
    if timeout is not None and time.monotonic() - started > timeout:
        return FORFEIT
    return outcome


@contextmanager
def _deadline(seconds: float | None) -> Iterator[None]:
    if seconds is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(*_: object) -> None:
        raise MoveTimeout

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
