import random
from collections.abc import Sequence
from pathlib import Path

from tienlen.engine import Play, PlayerView, Rules


class RandomBot:
    def __init__(self) -> None:
        self.rng = random.Random()

    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        self.rng.seed(seed)

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        return self.rng.choice(legal)


def create_bot(artifact_dir: Path | None) -> RandomBot:
    return RandomBot()
