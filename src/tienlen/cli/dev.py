import random
import time
from typing import Annotated

import typer

from tienlen.cli.options import PlayersOption, RuleOption, build_rules
from tienlen.cli.output import JsonOption, emit_record
from tienlen.engine import ENGINE_VERSION, Rules, apply, is_over, legal_plays, new_game, view

app = typer.Typer(help="Developer tools.", no_args_is_help=True)


@app.command(help="Measure engine throughput with random play.")
def bench(
    games: Annotated[int, typer.Option("--games", help="Games to play.")] = 500,
    players: PlayersOption = None,
    rule: RuleOption = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed.")] = 0,
    json_output: JsonOption = False,
) -> None:
    rules = build_rules(players, rule)
    rng = random.Random(seed)
    started = time.perf_counter()
    moves = sum(_random_game(rules, rng) for _ in range(games))
    elapsed = time.perf_counter() - started
    record = {
        "games": games,
        "n_players": rules.n_players,
        "seconds": elapsed,
        "games_per_second": games / elapsed,
        "moves_per_second": moves / elapsed,
    }
    emit_record(record, json_output)


@app.command(help="Print a rules config and its rules_id.")
def rules(
    players: PlayersOption = None,
    rule: RuleOption = None,
    json_output: JsonOption = False,
) -> None:
    config = build_rules(players, rule)
    record = {"rules_id": config.rules_id, "engine_version": ENGINE_VERSION, **config.to_dict()}
    emit_record(record, json_output)


def _random_game(rules: Rules, rng: random.Random) -> int:
    state = new_game(rules, rng.getrandbits(63))
    while not is_over(state):
        view(state, state.actor)
        state = apply(state, rng.choice(legal_plays(state)))
    return len(state.moves)
