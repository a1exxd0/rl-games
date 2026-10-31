import shutil
from pathlib import Path
from typing import Annotated

import typer

from tienlen.arena import validate_bot
from tienlen.bots.template import render_bot
from tienlen.cli.options import (
    DEFAULT_MOVE_TIMEOUT,
    DEFAULT_TEST_GAMES,
    DEFAULT_WORKERS,
    RuleOption,
    SeedOption,
    TimeoutOption,
    WorkersOption,
    build_rules,
    open_registry,
)
from tienlen.cli.output import JsonOption, emit_record, emit_rows
from tienlen.cli.testing import (
    WIN_RATE_COLUMNS,
    RunTestOption,
    TestGamesOption,
    emit_tested,
    test_version,
)
from tienlen.errors import RegistryError

app = typer.Typer(help="Create, validate, and version bots.", no_args_is_help=True)

VERSION_COLUMNS = (
    "bot",
    "version",
    "kind",
    "status",
    "browser_compatible",
    "rules_id",
    "parent_version",
)
LIST_COLUMNS = ("bot", "versions", "latest", "kind", "status", "browser_compatible")
RefArgument = Annotated[str, typer.Argument(help="Bot ref: name, name@N, name@latest, name@best.")]
ArtifactOption = Annotated[
    Path | None, typer.Option("--artifact-dir", help="Directory of weights or other files.")
]


@app.command(help="Create bots/<name>/bot.py from the template.")
def new(
    name: str,
    directory: Annotated[Path, typer.Option("--dir", help="Parent directory.")] = Path("bots"),
    json_output: JsonOption = False,
) -> None:
    target = directory / name / "bot.py"
    if target.exists():
        raise RegistryError(f"already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_bot(name))
    emit_record({"bot": name, "path": str(target)}, json_output)


@app.command(help="Load a bot file or ref and play it through short games.")
def validate(
    target: Annotated[str, typer.Argument(help="Path to a bot file, or a bot ref.")],
    artifact_dir: ArtifactOption = None,
    json_output: JsonOption = False,
) -> None:
    path, artifacts = _locate(target, artifact_dir)
    from tienlen.registry import browser_compatible, declared_dependencies

    result = validate_bot(path, artifacts)
    record = {
        "path": str(path),
        "ok": result.ok,
        "games": result.games,
        "problems": list(result.problems),
        "dependencies": declared_dependencies(path.read_text()),
        "browser_compatible": browser_compatible(path),
    }
    emit_record(record, json_output)
    if not result.ok:
        raise typer.Exit(1)


@app.command(help="Store a bot file as a new immutable version.")
def register(
    path: Path,
    name: Annotated[str | None, typer.Option("--name", help="Defaults to the folder.")] = None,
    kind: Annotated[str, typer.Option("--kind", help="rule, greedy, tree, nn, ...")] = "rule",
    description: Annotated[str, typer.Option("--description", help="Bot summary.")] = "",
    artifact_dir: ArtifactOption = None,
    parent: Annotated[str | None, typer.Option("--parent", help="Ref of the parent.")] = None,
    train_run: Annotated[str | None, typer.Option("--train-run", help="Run id.")] = None,
    rule: RuleOption = None,
    check: Annotated[bool, typer.Option("--check/--no-check", help="Validate first.")] = True,
    run_test: RunTestOption = True,
    test_games: TestGamesOption = DEFAULT_TEST_GAMES,
    workers: WorkersOption = DEFAULT_WORKERS,
    json_output: JsonOption = False,
) -> None:
    if not path.is_file():
        raise RegistryError(f"no such file: {path}")
    if check:
        result = validate_bot(path, artifact_dir)
        if not result.ok:
            raise RegistryError("; ".join(result.problems))
    registry = open_registry()
    version = registry.register(
        path,
        kind=kind,
        artifact_dir=artifact_dir,
        parent=parent,
        train_run_id=train_run,
        rules=build_rules(None, rule),
        name=name,
        description=description,
    )
    rows = test_version(registry, version, test_games, workers, rule) if run_test else []
    emit_tested({"ref": version.ref, "code_sha": version.code_sha}, rows, json_output)


@app.command("test", help="Play a bot against every other bot and show its win rates.")
def test_bot(
    ref: RefArgument,
    games: Annotated[
        int, typer.Option("--games", help="Games against each other bot.")
    ] = DEFAULT_TEST_GAMES,
    seed: SeedOption = None,
    rule: RuleOption = None,
    workers: WorkersOption = DEFAULT_WORKERS,
    move_timeout: TimeoutOption = DEFAULT_MOVE_TIMEOUT,
    json_output: JsonOption = False,
) -> None:
    registry = open_registry()
    version = registry.resolve(ref)
    rows = test_version(registry, version, games, workers, rule, seed, move_timeout)
    emit_rows(rows, WIN_RATE_COLUMNS, json_output)


@app.command("list", help="List bots with their latest version.")
def list_bots(json_output: JsonOption = False) -> None:
    emit_rows(_summaries(open_registry().version_rows()), LIST_COLUMNS, json_output)


@app.command(help="Show one bot version, its dependencies, and its ratings.")
def show(ref: RefArgument, json_output: JsonOption = False) -> None:
    from tienlen.registry import declared_dependencies

    registry = open_registry()
    version = registry.resolve(ref)
    row = next(
        row for row in registry.version_rows(version.name) if row["version"] == version.version
    )
    ratings = [
        {key: rating[key] for key in ("rules_id", "n_players", "mu", "sigma", "games")}
        for rating in registry.leaderboard()
        if (rating["bot"], rating["version"]) == (version.name, version.version)
    ]
    record = {
        **row,
        "path": str(version.path),
        "artifact_dir": str(version.artifact_dir) if version.artifact_dir else None,
        "dependencies": declared_dependencies(version.path.read_text()),
        "ratings": ratings,
    }
    emit_record(record, json_output)


@app.command(help="List every version of a bot.")
def versions(name: str, json_output: JsonOption = False) -> None:
    rows = open_registry().version_rows(name)
    if not rows:
        raise RegistryError(f"unknown bot: {name}")
    emit_rows(rows, VERSION_COLUMNS, json_output)


@app.command(help="Exclude a version from name@latest and name@best.")
def retire(ref: RefArgument, json_output: JsonOption = False) -> None:
    version = open_registry().retire(ref)
    emit_record({"ref": version.ref, "status": "retired"}, json_output)


@app.command(help="Copy a version's code and artifacts into a directory.")
def export(
    ref: RefArgument,
    out: Annotated[Path, typer.Option("--out", help="Target directory.")],
    json_output: JsonOption = False,
) -> None:
    version = open_registry().resolve(ref)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(version.path, out / "bot.py")
    if version.artifact_dir:
        shutil.copytree(version.artifact_dir, out / "artifacts", dirs_exist_ok=True)
    emit_record({"ref": version.ref, "out": str(out)}, json_output)


def _locate(target: str, artifact_dir: Path | None) -> tuple[Path, Path | None]:
    path = Path(target)
    if path.is_file():
        return path, artifact_dir
    version = open_registry().resolve(target)
    return version.path, artifact_dir or version.artifact_dir


def _summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["bot"], []).append(row)
    return [_summary(name, versions) for name, versions in grouped.items()]


def _summary(name: str, versions: list[dict]) -> dict:
    latest = versions[-1]
    return {
        "bot": name,
        "versions": len(versions),
        "latest": latest["version"],
        "kind": latest["kind"],
        "status": latest["status"],
        "browser_compatible": latest["browser_compatible"],
    }
