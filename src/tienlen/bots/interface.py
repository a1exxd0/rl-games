"""The contract every bot file meets.

A bot is a Python file that defines `create_bot(artifact_dir)` and returns a
`TienLenBot`. `artifact_dir` holds the bot version's stored files, such as
weights, and is None for bots that have none.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from tienlen.engine import Play, PlayerView, Rules


@runtime_checkable
class TienLenBot(Protocol):
    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        """Prepare for a new game. `seed` is for the bot's own RNG, never the deal seed."""
        ...

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        """Choose one of `legal`. It is never empty, and holds PASS when passing is allowed."""
        ...


class BotFactory(Protocol):
    def __call__(self, artifact_dir: Path | None) -> TienLenBot: ...
