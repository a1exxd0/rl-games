import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from openskill.models import PlackettLuce, PlackettLuceRating

Z_95 = 1.96
MAX_PASSES = 32
MAX_RATED_UPDATES = 500_000
SEAT_REF = re.compile(r"(?P<bot>.+)@(?P<version>\d+)")


@dataclass(frozen=True)
class Rating:
    bot: str
    version: int
    mu: float
    sigma: float
    games: int

    @property
    def ordinal(self) -> float:
        return self.mu - 3 * self.sigma


@dataclass(frozen=True)
class HeadToHead:
    wins: int
    losses: int
    ties: int

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def rate(self) -> float:
        return self.wins / self.decided if self.decided else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.wins, self.decided)


def is_rated(seats: Sequence[str]) -> bool:
    return len(set(seats)) > 1 and all(SEAT_REF.fullmatch(seat) for seat in seats)


def compute_ratings(games: Iterable[tuple[Sequence[str], Sequence[int]]]) -> list[Rating]:
    rated = sorted((tuple(seats), tuple(order)) for seats, order in games if is_rated(seats))
    played: Counter[str] = Counter(ref for seats, _ in rated for ref in set(seats))
    rng = random.Random(0)
    passes = [_rate_pass(rng.sample(rated, len(rated))) for _ in range(_pass_count(len(rated)))]
    return [_rating(ref, passes, games) for ref, games in played.items()]


def head_to_head(
    games: Iterable[tuple[Sequence[str], Sequence[int]]], first: str, second: str
) -> HeadToHead:
    tally = {-1: 0, 0: 0, 1: 0}
    for seats, order in games:
        if first in seats and second in seats:
            tally[_compare(seats, order, first, second)] += 1
    return HeadToHead(wins=tally[1], losses=tally[-1], ties=tally[0])


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if not trials:
        return (0.0, 1.0)
    rate = successes / trials
    denominator = 1 + Z_95**2 / trials
    centre = (rate + Z_95**2 / (2 * trials)) / denominator
    spread = Z_95 * math.sqrt(rate * (1 - rate) / trials + Z_95**2 / (4 * trials**2))
    return (max(0.0, centre - spread / denominator), min(1.0, centre + spread / denominator))


def _pass_count(games: int) -> int:
    return max(1, min(MAX_PASSES, MAX_RATED_UPDATES // max(games, 1)))


def _rate_pass(
    games: Sequence[tuple[Sequence[str], Sequence[int]]],
) -> dict[str, PlackettLuceRating]:
    model = PlackettLuce()
    players: dict[str, PlackettLuceRating] = defaultdict(model.rating)
    for seats, order in games:
        _rate_game(model, players, seats, order)
    return players


def _rate_game(
    model: PlackettLuce,
    players: Mapping[str, PlackettLuceRating],
    seats: Sequence[str],
    order: Sequence[int],
) -> None:
    ranks = _ranks(order)
    teams = [[players[ref]] for ref in seats]
    rated = model.rate(teams, ranks=[ranks[seat] for seat in range(len(seats))])
    updated = defaultdict(list)
    for ref, [player] in zip(seats, rated):
        updated[ref].append(player)
    for ref, copies in updated.items():
        players[ref].mu = fmean(copy.mu for copy in copies)
        players[ref].sigma = fmean(copy.sigma for copy in copies)


def _ranks(order: Sequence[int]) -> dict[int, int]:
    return {seat: position for position, seat in enumerate(order)}


def _compare(seats: Sequence[str], order: Sequence[int], first: str, second: str) -> int:
    ranks = _ranks(order)
    gap = _mean_rank(seats, ranks, second) - _mean_rank(seats, ranks, first)
    return (gap > 0) - (gap < 0)


def _mean_rank(seats: Sequence[str], ranks: Mapping[int, int], ref: str) -> float:
    return fmean(ranks[seat] for seat, held in enumerate(seats) if held == ref)


def _rating(ref: str, passes: Sequence[Mapping[str, PlackettLuceRating]], games: int) -> Rating:
    match = SEAT_REF.fullmatch(ref)
    if match is None:
        raise ValueError(f"not a seat ref: {ref}")
    mu = fmean(players[ref].mu for players in passes)
    sigma = fmean(players[ref].sigma for players in passes)
    return Rating(match["bot"], int(match["version"]), mu, sigma, games)
