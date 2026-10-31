import json
import os
import secrets
from typing import TYPE_CHECKING, Annotated, Any

import typer

from tienlen.engine import Rules

if TYPE_CHECKING:
    from tienlen.registry import Registry

PlayersOption = Annotated[int | None, typer.Option("--players", help="Seats per game, 2-4.")]
RuleOption = Annotated[
    list[str] | None,
    typer.Option("--rule", help="Rules override as key=value, e.g. lock_on_pass=false."),
]
SeedOption = Annotated[int | None, typer.Option("--seed", help="Schedule seed. Random if omitted.")]
TimeoutOption = Annotated[
    float, typer.Option("--move-timeout", help="Seconds per move before a forfeit.")
]
WorkersOption = Annotated[
    int, typer.Option("--workers", help="Worker processes. Defaults to all CPUs.")
]
DEFAULT_WORKERS = os.cpu_count() or 1
DEFAULT_MOVE_TIMEOUT = 1.0
DEFAULT_TEST_GAMES = 10_000
TEST_PLAYERS = 2


def open_registry() -> "Registry":
    from tienlen.registry import Registry

    return Registry.open()


def build_rules(players: int | None, overrides: list[str] | None) -> Rules:
    fields = dict(_parse_override(override) for override in overrides or [])
    if players is not None:
        fields["n_players"] = players
    try:
        return Rules(**fields)
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


def rules_filter(players: int | None, overrides: list[str] | None) -> tuple[str | None, int | None]:
    if players is None and not overrides:
        return None, None
    rules = build_rules(players, overrides)
    return rules.rules_id, rules.n_players


def pick_seed(seed: int | None) -> int:
    return secrets.randbits(63) if seed is None else seed


def _parse_override(override: str) -> tuple[str, Any]:
    key, separator, value = override.partition("=")
    if not separator:
        raise typer.BadParameter(f"expected key=value: {override}")
    try:
        return key, json.loads(value.lower())
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"invalid value: {override}") from error
