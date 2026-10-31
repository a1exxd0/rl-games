from dataclasses import dataclass
from pathlib import Path

from tienlen.arena.match import play_match
from tienlen.bots.loader import load_bot
from tienlen.engine import Rules
from tienlen.errors import BotLoadError

VALIDATION_SEEDS = (1, 2, 3)
VALIDATION_TIMEOUT = 5.0


@dataclass(frozen=True)
class Validation:
    games: int
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_bot(path: Path, artifact_dir: Path | None = None) -> Validation:
    problems = []
    games = 0
    for n_players in range(2, 5):
        rules = Rules(n_players=n_players)
        try:
            bots = [load_bot(path, artifact_dir) for _ in range(n_players)]
        except BotLoadError as error:
            return Validation(games, (str(error),))
        for seed in VALIDATION_SEEDS:
            result = play_match(bots, rules, seed, VALIDATION_TIMEOUT)
            games += 1
            if result.forfeits:
                problems.append(f"{n_players}-player seed {seed}: forfeited {result.forfeits}")
    return Validation(games, tuple(problems))
