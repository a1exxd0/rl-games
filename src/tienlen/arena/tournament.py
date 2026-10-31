import random
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import combinations, count, islice
from multiprocessing import get_context
from pathlib import Path

from tienlen.arena.match import MatchResult, play_match
from tienlen.bots.interface import TienLenBot
from tienlen.bots.loader import load_bot
from tienlen.engine import Rules

CHUNKS_PER_WORKER = 8
_loaded: dict[tuple[str, int], TienLenBot] = {}


@dataclass(frozen=True)
class Entrant:
    ref: str
    path: Path
    artifact_dir: Path | None = None


@dataclass(frozen=True)
class Fixture:
    seats: tuple[Entrant, ...]
    seed: int


def schedule(entrants: Sequence[Entrant], n_players: int, games: int, seed: int) -> list[Fixture]:
    tables = _tables(entrants, n_players)
    rng = random.Random(seed)
    fixtures = (
        Fixture(_rotated(tables[index % len(tables)], turn), deal)
        for index in count()
        for deal in [rng.getrandbits(63)]
        for turn in range(n_players)
    )
    return list(islice(fixtures, games))


def play_fixture(fixture: Fixture, rules: Rules, move_timeout: float | None) -> MatchResult:
    bots = [_bot(entrant, seat) for seat, entrant in enumerate(fixture.seats)]
    seats = [entrant.ref for entrant in fixture.seats]
    return play_match(bots, rules, fixture.seed, move_timeout, seats)


def fixed_fixtures(seats: Sequence[Entrant], games: int, seed: int) -> list[Fixture]:
    rng = random.Random(seed)
    return [Fixture(tuple(seats), rng.getrandbits(63)) for _ in range(games)]


def run_fixtures(
    fixtures: Sequence[Fixture],
    rules: Rules,
    workers: int = 1,
    move_timeout: float | None = None,
) -> Iterator[MatchResult]:
    play = partial(play_fixture, rules=rules, move_timeout=move_timeout)
    if workers <= 1:
        yield from map(play, fixtures)
        return
    chunksize = max(1, len(fixtures) // (workers * CHUNKS_PER_WORKER))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        yield from pool.map(play, fixtures, chunksize=chunksize)


def run_arena(
    entrants: Sequence[Entrant],
    rules: Rules,
    games: int,
    seed: int,
    workers: int = 1,
    move_timeout: float | None = None,
) -> Iterator[MatchResult]:
    fixtures = schedule(entrants, rules.n_players, games, seed)
    return run_fixtures(fixtures, rules, workers, move_timeout)


def _tables(entrants: Sequence[Entrant], n_players: int) -> list[tuple[Entrant, ...]]:
    if not entrants:
        raise ValueError("an arena needs at least one entrant")
    if len(entrants) >= n_players:
        return list(combinations(entrants, n_players))
    return [tuple(entrants[seat % len(entrants)] for seat in range(n_players))]


def _rotated(table: tuple[Entrant, ...], turn: int) -> tuple[Entrant, ...]:
    return table[turn:] + table[:turn]


def _bot(entrant: Entrant, seat: int) -> TienLenBot:
    key = (entrant.ref, seat)
    if key not in _loaded:
        _loaded[key] = load_bot(entrant.path, entrant.artifact_dir)
    return _loaded[key]
