import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from tienlen.engine import Rules
from tienlen.rl.league import from_refs
from tienlen.rl.runs import RunDir, data_root, iter_runs, runs_root
from tienlen.rl.trainers import get_trainer

app = typer.Typer(help="Train bots.", no_args_is_help=True)

JsonOption = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
BOT_KIND = "nn"


def _open_registry():
    from tienlen.registry import Registry

    return Registry.open()


def _emit(payload: Any, as_json: bool, text: str) -> None:
    typer.echo(json.dumps(payload, indent=2) if as_json else text)


def _find_run(run_id: str) -> RunDir:
    run = RunDir(runs_root() / run_id)
    if not run.config_path.exists():
        typer.echo(f"no such run: {run_id}", err=True)
        raise typer.Exit(1)
    return run


def _parse_override(item: str) -> tuple[str, Any]:
    key, separator, raw = item.partition("=")
    if not separator:
        raise typer.BadParameter(f"expected key=value, got {item!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def _bot_path(config: dict[str, Any]) -> Path:
    return Path("bots") / config["bot"] / "bot.py"


def _register_snapshots(registry: Any, run: RunDir, config: dict[str, Any]):
    rules = Rules.from_dict(config["rules"])

    def on_snapshot(iteration: int, snapshot: Path) -> None:
        version = registry.register(
            _bot_path(config),
            kind=BOT_KIND,
            artifact_dir=snapshot,
            parent=run.parent(),
            train_run_id=run.run_id,
            rules=rules,
            name=config["bot"],
        )
        run.set_parent(version.ref)
        typer.echo(f"iteration {iteration}: registered {version.ref}", err=True)

    return on_snapshot


def _execute(registry: Any, run: RunDir, config: dict[str, Any]) -> str:
    opponents = from_refs(config["opponents"], registry.resolve)
    trainer = get_trainer(config["trainer"])
    run.clear_stop()
    run.set_status("running")
    try:
        status = trainer(config, run, opponents, _register_snapshots(registry, run, config))
    except KeyboardInterrupt:
        status = "stopped"
    except Exception:
        _finish(registry, run, "failed")
        raise
    _finish(registry, run, status)
    return status


def _finish(registry: Any, run: RunDir, status: str) -> None:
    run.set_status(status)
    registry.finish_run(run.run_id, status, metrics_uri=str(run.metrics_path.resolve()))


def _summary(run: RunDir) -> dict[str, Any]:
    metrics = run.metrics()
    config = run.config()
    return {
        "run_id": run.run_id,
        "bot": config["bot"],
        "trainer": config.get("trainer", "lab"),
        "status": run.status(),
        "iteration": metrics[-1]["iteration"] if metrics else 0,
        "iterations": config["iterations"],
        "opponents": metrics[-1].get("opponents", {}) if metrics else {},
    }


def _summary_line(summary: dict[str, Any]) -> str:
    progress = f"{summary['iteration']}/{summary['iterations']}"
    return f"{summary['run_id']}  {summary['status']}  {summary['bot']}  {progress}"


def _start_config(
    bot: str,
    trainer: str,
    players: int,
    opponents: str,
    options: dict[str, Any],
    overrides: Sequence[str],
) -> dict[str, Any]:
    return {
        "bot": bot,
        "trainer": trainer,
        "rules": Rules(n_players=players).to_dict(),
        "opponents": [ref.strip() for ref in opponents.split(",") if ref.strip()],
        **options,
        **dict(_parse_override(item) for item in overrides),
    }


@app.command(help="Start a training run in the foreground.")
def start(
    bot: Annotated[str, typer.Option(help="Bot folder under bots/.")] = "neural",
    trainer: Annotated[str, typer.Option(help="Trainer name.")] = "ppo",
    players: Annotated[int, typer.Option(help="Players per game, 2-4.")] = 2,
    opponents: Annotated[
        str, typer.Option(help="Comma-separated bot refs for the league.")
    ] = "random@latest,greedy@latest",
    iterations: Annotated[int, typer.Option(help="Training iterations.")] = 50,
    episodes: Annotated[int, typer.Option(help="Episodes per iteration.")] = 2000,
    workers: Annotated[int, typer.Option(help="Collection processes.")] = DEFAULT_WORKERS,
    snapshot_every: Annotated[int, typer.Option(help="Iterations between snapshots.")] = 10,
    seed: Annotated[int, typer.Option(help="Training seed.")] = 0,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Trainer setting, as key=value.")
    ] = None,
    as_json: JsonOption = False,
) -> None:
    config = _start_config(
        bot,
        trainer,
        players,
        opponents,
        {
            "iterations": iterations,
            "episodes": episodes,
            "workers": workers,
            "snapshot_every": snapshot_every,
            "seed": seed,
        },
        overrides or [],
    )
    registry = _open_registry()
    run = RunDir(runs_root() / registry.start_run(bot, config))
    run.create(config)
    _execute(registry, run, config)
    summary = _summary(run)
    _emit(summary, as_json, _summary_line(summary))


@app.command(help="Continue a run from its last checkpoint.")
def resume(
    run_id: Annotated[str, typer.Argument(help="Run to continue.")],
    iterations: Annotated[int | None, typer.Option(help="New total iterations.")] = None,
    as_json: JsonOption = False,
) -> None:
    run = _find_run(run_id)
    config = run.config()
    if iterations is not None:
        config["iterations"] = iterations
        run.config_path.write_text(json.dumps(config, indent=2))
    _execute(_open_registry(), run, config)
    summary = _summary(run)
    _emit(summary, as_json, _summary_line(summary))


@app.command("list", help="List training runs.")
def list_runs(as_json: JsonOption = False) -> None:
    summaries = [_summary(run) for run in iter_runs(data_root())]
    text = "\n".join(_summary_line(summary) for summary in summaries) or "no runs"
    _emit(summaries, as_json, text)


@app.command(help="Show a run's status and latest win rates.")
def show(
    run_id: Annotated[str, typer.Argument(help="Run to show.")],
    as_json: JsonOption = False,
) -> None:
    summary = _summary(_find_run(run_id))
    rates = [
        f"  {name}: {stats['win_rate']:.3f} over {stats['games']} games"
        for name, stats in summary["opponents"].items()
    ]
    _emit(summary, as_json, "\n".join([_summary_line(summary), *rates]))


@app.command(help="Ask a running trainer to stop after this iteration.")
def stop(
    run_id: Annotated[str, typer.Argument(help="Run to stop.")],
    as_json: JsonOption = False,
) -> None:
    run = _find_run(run_id)
    run.request_stop()
    _emit({"run_id": run_id, "stop_requested": True}, as_json, f"stop requested: {run_id}")
