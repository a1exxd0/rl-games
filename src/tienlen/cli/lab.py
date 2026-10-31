import json
from pathlib import Path
from typing import Annotated, Any

import typer

from tienlen.cli.options import (
    DEFAULT_TEST_GAMES,
    DEFAULT_WORKERS,
    WorkersOption,
    open_registry,
)
from tienlen.cli.output import JsonOption, emit_json, emit_record, emit_rows
from tienlen.cli.testing import RunTestOption, TestGamesOption, emit_tested, test_version
from tienlen.errors import RegistryError

app = typer.Typer(
    help="Grow models one change at a time and record what helped.", no_args_is_help=True
)

LOG_COLUMNS = (
    "step",
    "bot_ref",
    "parent_ref",
    "verdict",
    "vs_random",
    "vs_greedy",
    "h2h",
    "change",
)
DEFAULT_REPORT = Path("docs/evolution.md")


def _parse_override(item: str) -> tuple[str, Any]:
    key, separator, raw = item.partition("=")
    if not separator:
        raise typer.BadParameter(f"expected key=value, got {item!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def _mean(results: dict[str, Any], role: str) -> float | None:
    entry = results.get("summary", {}).get(role)
    return entry["mean"] if entry else None


def _log_row(row: dict[str, Any]) -> dict[str, Any]:
    results = json.loads(row["results"]) if row["results"] else {}
    head_to_head = results.get("head_to_head") or {}
    return {
        "step": row["step"],
        "bot_ref": row["bot_ref"],
        "parent_ref": row["parent_ref"],
        "verdict": row["verdict"],
        "vs_random": _mean(results, "random"),
        "vs_greedy": _mean(results, "greedy"),
        "h2h": head_to_head.get("win_rate"),
        "change": row["change"],
    }


def _experiment_record(row: dict[str, Any]) -> dict[str, Any]:
    decoded = {key: json.loads(row[key]) if row[key] else None for key in ("config", "results")}
    return {**row, **decoded, "created_at": str(row["created_at"])}


@app.command(help="Train a step's seeds, run the fixed suite, and decide its verdict.")
def step(
    folder: Annotated[str, typer.Argument(help="Step folder under bots/, like lab_02_mlp.")],
    change: Annotated[str, typer.Option("--change", help="The one change in this step.")],
    hypothesis: Annotated[
        str, typer.Option("--hypothesis", help="Expected effect, recorded before training.")
    ],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Training setting, as key=value.")
    ] = None,
    budget_change: Annotated[
        bool, typer.Option("--budget-change", help="Allow this step to change the game budget.")
    ] = False,
    parent: Annotated[
        str | None, typer.Option("--parent", help="Parent ref. Defaults to the current best.")
    ] = None,
    workers: WorkersOption = DEFAULT_WORKERS,
    run_test: RunTestOption = True,
    test_games: TestGamesOption = DEFAULT_TEST_GAMES,
    json_output: JsonOption = False,
) -> None:
    from tienlen.lab.step import StepRequest, run_step

    request = StepRequest(
        folder,
        change,
        hypothesis,
        dict(_parse_override(item) for item in overrides or []),
        budget_change,
        parent,
        workers,
    )
    registry = open_registry()
    row = run_step(registry, request)
    version = registry.resolve(row["bot_ref"]) if run_test else None
    rows = test_version(registry, version, test_games, workers) if version else []
    emit_tested(_log_row(row), rows, json_output)


@app.command(help="List every step and its verdict.")
def log(json_output: JsonOption = False) -> None:
    rows = [_log_row(row) for row in open_registry().experiments()]
    emit_rows(rows, LOG_COLUMNS, json_output)


@app.command(help="Show one step's full record.")
def show(
    number: Annotated[int, typer.Argument(help="Step number.")],
    json_output: JsonOption = False,
) -> None:
    record = _experiment_record(open_registry().experiment(number))
    if json_output:
        emit_json(record)
        return
    emit_record({key: _text(value) for key, value in record.items()}, False)


def _text(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, dict) else value


@app.command(help="Attach a note on what a step showed.")
def note(
    number: Annotated[int, typer.Argument(help="Step number.")],
    text: Annotated[str, typer.Argument(help="What the step showed.")],
    json_output: JsonOption = False,
) -> None:
    registry = open_registry()
    updated = registry.update_experiment(registry.experiment(number)["experiment_id"], notes=text)
    emit_record(_log_row(updated), json_output)


@app.command(help="Run the fixed suite on an outside bot, for comparison.")
def reference(
    ref: Annotated[str, typer.Argument(help="Bot ref to measure.")],
    workers: WorkersOption = DEFAULT_WORKERS,
    json_output: JsonOption = False,
) -> None:
    from tienlen.lab.step import run_reference

    row = run_reference(open_registry(), ref, workers=workers)
    if json_output:
        emit_json(_experiment_record({**row, "config": None, "notes": None}))
        return
    typer.echo(row["results"])


def _ratings(registry: Any) -> dict[str, float]:
    return {f"{row['bot']}@{row['version']}": row["ordinal"] for row in registry.leaderboard()}


@app.command(help="Write the evolution page from the experiments table.")
def report(
    output: Annotated[Path, typer.Option("--output", help="File to write.")] = DEFAULT_REPORT,
    json_output: JsonOption = False,
) -> None:
    from tienlen.lab.report import render

    registry = open_registry()
    rows = registry.experiments()
    if not rows:
        raise RegistryError("no experiments to report")
    output.write_text(render(rows, _ratings(registry)))
    emit_record({"output": str(output), "steps": len(rows)}, json_output)
