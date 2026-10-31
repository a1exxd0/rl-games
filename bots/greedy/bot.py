from collections.abc import Sequence
from pathlib import Path

from tienlen.engine import PASS, Play, PlayerView, Rules


class GreedyBot:
    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        pass

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        plays = [play for play in legal if not play.is_pass]
        return min(plays, key=lambda play: (play.top, len(play.cards)), default=PASS)


def create_bot(artifact_dir: Path | None) -> GreedyBot:
    return GreedyBot()
