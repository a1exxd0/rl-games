import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Self

MIN_PLAYERS = 2
MAX_PLAYERS = 4


@dataclass(frozen=True, slots=True)
class Rules:
    n_players: int = 4
    lock_on_pass: bool = True
    chop_after_pass: bool = False
    instant_wins: bool = False
    first_lead_includes_lowest: bool = True
    forbid_finish_on_two: bool = False

    def __post_init__(self) -> None:
        if not MIN_PLAYERS <= self.n_players <= MAX_PLAYERS:
            raise ValueError(f"n_players must be {MIN_PLAYERS}-{MAX_PLAYERS}: {self.n_players}")

    @property
    def rules_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(**data)
