from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from tienlen.arena import Entrant, MatchResult, schedule, wilson_interval
from tienlen.arena.tournament import run_fixtures
from tienlen.engine import Rules

SUITE_GAMES = 4000
SUITE_SEED = 20261031
SUITE_RULES = Rules(n_players=2)
RANDOM_ROLE = "random"
GREEDY_ROLE = "greedy"
BEST_ROLE = "best"


@dataclass(frozen=True, slots=True)
class Pairing:
    opponent: str
    wins: int
    games: int
    win_rate: float
    ci_low: float
    ci_high: float
    forfeits: int

    def to_dict(self) -> dict:
        return asdict(self)


def fixtures_for(subject: Entrant, opponent: Entrant, games: int) -> list:
    return schedule([subject, opponent], SUITE_RULES.n_players, games, SUITE_SEED)


def _pairing(subject: str, opponent: str, results: Iterable[MatchResult]) -> Pairing:
    played = list(results)
    wins = sum(result.seats[result.placements[0]] == subject for result in played)
    low, high = wilson_interval(wins, len(played))
    forfeits = sum(len(result.forfeits) for result in played)
    return Pairing(opponent, wins, len(played), wins / len(played), low, high, forfeits)


def run_suite(
    subjects: Sequence[Entrant],
    opponents: Mapping[str, Entrant],
    *,
    workers: int,
    games: int = SUITE_GAMES,
) -> dict[str, dict[str, Pairing]]:
    jobs = [(subject, role, entrant) for subject in subjects for role, entrant in opponents.items()]
    fixtures = [
        fixture for subject, _, entrant in jobs for fixture in fixtures_for(subject, entrant, games)
    ]
    results = iter(run_fixtures(fixtures, SUITE_RULES, workers))
    table: dict[str, dict[str, Pairing]] = {subject.ref: {} for subject in subjects}
    for subject, role, entrant in jobs:
        played = [next(results) for _ in range(games)]
        table[subject.ref][role] = _pairing(subject.ref, entrant.ref, played)
    return table
