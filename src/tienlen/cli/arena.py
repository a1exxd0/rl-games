import uuid
from itertools import batched
from typing import Annotated

import typer

from tienlen.arena import MatchResult, run_arena
from tienlen.cli.options import (
    DEFAULT_MOVE_TIMEOUT,
    DEFAULT_WORKERS,
    PlayersOption,
    RuleOption,
    SeedOption,
    TimeoutOption,
    WorkersOption,
    build_rules,
    open_registry,
    pick_seed,
    rules_filter,
)
from tienlen.cli.output import JsonOption, emit_json, emit_rows
from tienlen.engine.rules import MAX_PLAYERS, MIN_PLAYERS

app = typer.Typer(help="Run tournaments and compare bots.", no_args_is_help=True)

STANDING_COLUMNS = ("ref", "games", "wins", "win_rate", "mean_rank", "forfeits")
BOARD_COLUMNS = ("bot", "version", "rules_id", "n_players", "ordinal", "mu", "sigma", "games")
H2H_COLUMNS = (
    "first",
    "second",
    "n_players",
    "wins",
    "losses",
    "ties",
    "win_rate",
    "ci_low",
    "ci_high",
)
DEFAULT_BATCH = 500


@app.command(help="Play rotated, seeded games, record them, and update ratings.")
def run(
    refs: Annotated[list[str], typer.Argument(help="Bot refs to enter.")],
    games: Annotated[int, typer.Option("--games", help="Games to play.")] = 100,
    seed: SeedOption = None,
    players: PlayersOption = None,
    rule: RuleOption = None,
    workers: WorkersOption = DEFAULT_WORKERS,
    move_timeout: TimeoutOption = DEFAULT_MOVE_TIMEOUT,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Games per commit.")
    ] = DEFAULT_BATCH,
    json_output: JsonOption = False,
) -> None:
    rules = build_rules(players or max(MIN_PLAYERS, min(len(refs), MAX_PLAYERS)), rule)
    registry = open_registry()
    entrants = [registry.resolve(ref, rules=rules).entrant for ref in refs]
    seed = pick_seed(seed)
    arena_id = uuid.uuid4().hex[:12]
    results = run_arena(entrants, rules, games, seed, min(workers, games), move_timeout)
    standings = _Standings()
    for batch in batched(results, batch_size):
        registry.record_matches(batch, arena_id=arena_id)
        standings.add(batch)
    registry.update_ratings(rules.rules_id, rules.n_players)
    rows = standings.rows()
    if json_output:
        emit_json(
            {"arena_id": arena_id, "seed": seed, "rules_id": rules.rules_id, "standings": rows}
        )
        return
    typer.echo(f"arena {arena_id}  seed {seed}  rules {rules.rules_id}")
    emit_rows(rows, STANDING_COLUMNS, False)


@app.command(help="Show ratings, best first. Filter with --players or --rule.")
def leaderboard(
    players: PlayersOption = None,
    rule: RuleOption = None,
    limit: Annotated[int, typer.Option("--limit", help="Rows to show.")] = 20,
    json_output: JsonOption = False,
) -> None:
    rows = open_registry().leaderboard(*rules_filter(players, rule))
    emit_rows(rows[:limit], BOARD_COLUMNS, json_output)


@app.command(help="Compare two refs over recorded games, with a 95% Wilson interval.")
def h2h(first: str, second: str, json_output: JsonOption = False) -> None:
    emit_rows(open_registry().head_to_head(first, second), H2H_COLUMNS, json_output)


class _Standings:
    def __init__(self) -> None:
        self.tally: dict[str, dict[str, float]] = {}

    def add(self, results: tuple[MatchResult, ...]) -> None:
        for result in results:
            for rank, seat in enumerate(result.placements):
                self._add_seat(result, rank, seat)

    def rows(self) -> list[dict]:
        rows = [_standing(ref, tally) for ref, tally in self.tally.items()]
        return sorted(rows, key=lambda row: row["mean_rank"])

    def _add_seat(self, result: MatchResult, rank: int, seat: int) -> None:
        tally = self.tally.setdefault(
            result.seats[seat], {"games": 0, "wins": 0, "ranks": 0, "forfeits": 0}
        )
        tally["games"] += 1
        tally["wins"] += rank == 0
        tally["ranks"] += rank
        tally["forfeits"] += seat in result.forfeits


def _standing(ref: str, tally: dict[str, float]) -> dict:
    return {
        "ref": ref,
        "games": tally["games"],
        "wins": tally["wins"],
        "win_rate": tally["wins"] / tally["games"],
        "mean_rank": tally["ranks"] / tally["games"],
        "forfeits": tally["forfeits"],
    }
