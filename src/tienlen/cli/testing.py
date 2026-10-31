from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import typer

from tienlen.cli.options import (
    DEFAULT_MOVE_TIMEOUT,
    DEFAULT_TEST_GAMES,
    DEFAULT_WORKERS,
    TEST_PLAYERS,
    build_rules,
    pick_seed,
)
from tienlen.cli.output import emit_json, emit_record, emit_rows

if TYPE_CHECKING:
    from tienlen.registry import BotVersion, Registry

WIN_RATE_COLUMNS = ("first", "second", "games", "win_rate", "ci_low", "ci_high")
TestGamesOption = Annotated[int, typer.Option("--test-games", help="Games against each other bot.")]
RunTestOption = Annotated[
    bool, typer.Option("--test/--no-test", help="Measure win rates against every other bot.")
]


def _announce(ref: str, games: int) -> None:
    typer.echo(f"playing {games} games against {ref}", err=True)


def test_version(
    registry: "Registry",
    version: "BotVersion",
    games: int = DEFAULT_TEST_GAMES,
    workers: int = DEFAULT_WORKERS,
    rule: list[str] | None = None,
    seed: int | None = None,
    move_timeout: float = DEFAULT_MOVE_TIMEOUT,
) -> list[dict[str, Any]]:
    from tienlen.registry import BotTest, run_bot_test

    rules = build_rules(TEST_PLAYERS, rule)
    test = BotTest(rules, pick_seed(seed), games, workers, move_timeout)
    return run_bot_test(registry, version, test, _announce)


def emit_tested(
    record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], as_json: bool
) -> None:
    if as_json:
        emit_json({**record, "win_rates": rows})
        return
    emit_record(record, False)
    emit_rows(rows, WIN_RATE_COLUMNS, False)
