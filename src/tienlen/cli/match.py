from itertools import batched
from typing import Annotated

import typer

from tienlen.arena import fixed_fixtures, run_fixtures
from tienlen.cli.options import (
    DEFAULT_MOVE_TIMEOUT,
    PlayersOption,
    RuleOption,
    SeedOption,
    TimeoutOption,
    build_rules,
    open_registry,
    pick_seed,
)
from tienlen.cli.output import JsonOption, emit_json, emit_record, emit_rows
from tienlen.engine import (
    IllegalPlay,
    Rules,
    format_cards,
    moves_from_list,
    new_game,
)
from tienlen.engine import (
    replay as replay_game,
)
from tienlen.errors import RegistryError

app = typer.Typer(help="Play, inspect, and replay single matches.", no_args_is_help=True)

MATCH_COLUMNS = ("match_id", "played_at", "seats", "placements", "forfeits", "turns")
BATCH_SIZE = 500


@app.command(help="Play fixed seats without seat rotation. Give one ref per seat.")
def run(
    refs: Annotated[list[str], typer.Argument(help="Bot refs in seat order.")],
    games: Annotated[int, typer.Option("--games", help="Games to play.")] = 1,
    seed: SeedOption = None,
    players: PlayersOption = None,
    rule: RuleOption = None,
    move_timeout: TimeoutOption = DEFAULT_MOVE_TIMEOUT,
    json_output: JsonOption = False,
) -> None:
    rules = build_rules(players or len(refs), rule)
    if len(refs) != rules.n_players:
        raise RegistryError(f"need {rules.n_players} refs, got {len(refs)}")
    registry = open_registry()
    seats = [registry.resolve(ref, rules=rules).entrant for ref in refs]
    seed = pick_seed(seed)
    results = run_fixtures(fixed_fixtures(seats, games, seed), rules, move_timeout=move_timeout)
    rows = []
    for batch in batched(results, BATCH_SIZE):
        ids = registry.record_matches(batch)
        rows += [_summary(match_id, result) for match_id, result in zip(ids, batch)]
    emit_rows(rows, ("match_id", "seats", "placements", "forfeits", "turns"), json_output)


@app.command("list", help="List recorded matches, newest first.")
def list_matches(
    arena: Annotated[str | None, typer.Option("--arena", help="Arena id.")] = None,
    bot: Annotated[str | None, typer.Option("--bot", help="Bot name or ref.")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Rows to show.")] = 20,
    json_output: JsonOption = False,
) -> None:
    rows = open_registry().matches(arena_id=arena, bot=bot, limit=limit)
    emit_rows(rows, MATCH_COLUMNS, json_output)


@app.command(help="Show one match by id or id prefix.")
def show(match_id: str, json_output: JsonOption = False) -> None:
    emit_record(open_registry().match(match_id), json_output)


@app.command(help="Verify a stored replay and print it move by move.")
def replay(match_id: str, json_output: JsonOption = False) -> None:
    stored = open_registry().replay(match_id)
    rules = Rules.from_dict(stored["rules"])
    moves = moves_from_list(stored["moves"])
    try:
        replay_game(rules, stored["seed"], moves)
    except IllegalPlay as error:
        raise RegistryError(f"replay does not fit the engine: {error}") from error
    if json_output:
        emit_json(stored)
    else:
        _print_replay(stored, rules, moves)


def _summary(match_id: str, result) -> dict:
    return {
        "match_id": match_id,
        "seats": list(result.seats),
        "placements": list(result.placements),
        "forfeits": list(result.forfeits),
        "turns": len(result.moves),
    }


def _print_replay(stored: dict, rules: Rules, moves) -> None:
    seats = stored["seats"]
    for seat, hand in enumerate(new_game(rules, stored["seed"]).hands):
        typer.echo(f"seat {seat} {seats[seat]}: {format_cards(hand)}")
    for move in moves:
        cards = format_cards(move.play.cards) if not move.play.is_pass else "pass"
        typer.echo(f"{move.seat}: {cards}")
    typer.echo("placements: " + " ".join(seats[seat] for seat in stored["placements"]))
