import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

CURRENT = "current"


@dataclass(frozen=True, slots=True)
class Opponent:
    name: str
    path: Path | None = None
    artifact_dir: Path | None = None

    @property
    def bot_file(self) -> Path:
        if self.path is None:
            raise ValueError(f"{self.name} has no bot file")
        return self.path


CURRENT_POLICY = Opponent(CURRENT)


class League:
    def __init__(
        self,
        fixed: Sequence[Opponent],
        *,
        self_share: float = 0.4,
        snapshot_share: float = 0.3,
        max_snapshots: int = 10,
    ) -> None:
        self.fixed = list(fixed)
        self.snapshots: list[Opponent] = []
        self.self_share = self_share
        self.snapshot_share = snapshot_share
        self.max_snapshots = max_snapshots

    def add_snapshot(self, opponent: Opponent) -> None:
        self.snapshots = [*self.snapshots, opponent][-self.max_snapshots :]

    def _pick(self, rng: random.Random) -> Opponent:
        roll = rng.random()
        if roll < self.self_share:
            return CURRENT_POLICY
        tier = self.snapshots if roll < self.self_share + self.snapshot_share else self.fixed
        return rng.choice(tier or self.fixed or [CURRENT_POLICY])

    def sample(self, rng: random.Random, count: int) -> list[Opponent]:
        return [self._pick(rng) for _ in range(count)]


def from_refs(refs: Iterable[str], resolve) -> list[Opponent]:
    versions = ((ref, resolve(ref)) for ref in refs)
    return [
        Opponent(ref, Path(v.path), v.artifact_dir and Path(v.artifact_dir)) for ref, v in versions
    ]
