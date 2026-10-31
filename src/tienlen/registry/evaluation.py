import uuid
from collections.abc import Callable
from dataclasses import dataclass
from itertools import batched

from tienlen.arena import run_arena
from tienlen.engine import Rules
from tienlen.registry.registry import RETIRED, BotVersion, Registry
from tienlen.registry.tables import Row

BATCH_SIZE = 2_000


@dataclass(frozen=True)
class BotTest:
    rules: Rules
    seed: int
    games: int
    workers: int = 1
    move_timeout: float | None = None


def rivals(registry: Registry, version: BotVersion) -> list[BotVersion]:
    chosen = {
        row["bot"]: row["version"] for row in registry.version_rows() if row["status"] != RETIRED
    }
    for row in registry.experiments():
        if row["bot_ref"]:
            name, _, number = row["bot_ref"].partition("@")
            chosen[name] = int(number)
    return [
        registry.resolve(f"{name}@{number}")
        for name, number in chosen.items()
        if (name, number) != (version.name, version.version)
    ]


def _play(registry: Registry, version: BotVersion, rival: BotVersion, games: int, test: BotTest):
    arena_id = uuid.uuid4().hex[:12]
    results = run_arena(
        [version.entrant, rival.entrant],
        test.rules,
        games,
        test.seed,
        min(test.workers, games),
        test.move_timeout,
    )
    for batch in batched(results, BATCH_SIZE):
        registry.record_matches(batch, arena_id=arena_id, replays=False)


def _recorded(registry: Registry, ref: str, rules: Rules) -> dict[str, int]:
    return {row["second"]: row["games"] for row in registry.win_rates(rules) if row["first"] == ref}


def run_bot_test(
    registry: Registry,
    version: BotVersion,
    test: BotTest,
    on_rival: Callable[[str, int], None] = lambda ref, games: None,
) -> list[Row]:
    opponents = rivals(registry, version)
    recorded = _recorded(registry, version.ref, test.rules)
    missing = {rival.ref: test.games - recorded.get(rival.ref, 0) for rival in opponents}
    for rival in opponents:
        if missing[rival.ref] > 0:
            on_rival(rival.ref, missing[rival.ref])
            _play(registry, version, rival, missing[rival.ref], test)
    if max(missing.values(), default=0) > 0:
        registry.update_ratings(test.rules.rules_id, test.rules.n_players)
    return [
        row
        for row in registry.win_rates(test.rules)
        if row["first"] == version.ref and row["second"] in missing
    ]
