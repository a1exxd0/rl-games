import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from tienlen.cli.lab import _log_row
from tienlen.cli.options import TEST_PLAYERS
from tienlen.engine import Rules

if TYPE_CHECKING:
    from tienlen.registry import Registry

ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = ROOT / "web"
BOTS_DIR = ROOT / "bots"
DIST = WEB_DIR / "dist"
BROWSER_PACKAGES = frozenset({"numpy"})
BROWSER_TIENLEN_MODULES = (
    "tienlen.engine",
    "tienlen.bots",
    "tienlen.rl.policy",
    "tienlen.rl.encoding",
)
NUMPY_TIENLEN_MODULES = ("tienlen.rl.",)

app = typer.Typer(help="Build and serve the browser app.", no_args_is_help=True)


def imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return modules


def browser_packages(bot_file: Path) -> list[str] | None:
    modules = imported_modules(bot_file.read_text())
    tops = {module.partition(".")[0] for module in modules}
    third_party = tops - sys.stdlib_module_names - {"tienlen"}
    tienlen_modules = {m for m in modules if m == "tienlen" or m.startswith("tienlen.")}
    if not third_party <= BROWSER_PACKAGES:
        return None
    if not all(m.startswith(BROWSER_TIENLEN_MODULES) for m in tienlen_modules):
        return None
    if any(m.startswith(NUMPY_TIENLEN_MODULES) for m in tienlen_modules):
        third_party.add("numpy")
    return sorted(third_party)


def visible_files(folder: Path) -> list[Path]:
    files = (path for path in folder.iterdir() if path.is_file())
    return [path for path in files if not path.name.startswith(".")]


def starts(bot_file: Path) -> bool:
    artifacts = [path for path in visible_files(bot_file.parent) if path != bot_file]
    spec = importlib.util.spec_from_file_location(f"web_{bot_file.parent.name}", bot_file)
    if spec is None or spec.loader is None:
        return False
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.create_bot(bot_file.parent if artifacts else None)
    except Exception:  # noqa: BLE001
        return False
    return True


def install_bot(name: str, files: list[Path], packages: list[str], out: Path) -> dict[str, Any]:
    target = out / "bots" / name
    target.mkdir(parents=True)
    for file in files:
        shutil.copy2(file, target / file.name)
    return {"name": name, "files": sorted(file.name for file in files), "packages": packages}


def build_wheel(out: Path) -> Path:
    wheels = out / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels), "--project", str(ROOT)],
        check=True,
        capture_output=True,
    )
    return next(wheels.glob("tienlen-*.whl")).relative_to(out)


def read_registry() -> "Registry | None":
    from tienlen.registry import Registry
    from tienlen.registry.tables import TableStore

    root = Path(os.environ.get("TIENLEN_DATA", "data")).resolve()
    if not (root / "catalog.db").exists():
        return None
    return Registry(root, TableStore(root))


def read_experiments(registry: "Registry | None") -> list[dict[str, Any]]:
    from tienlen.errors import RegistryError

    if registry is None:
        return []
    try:
        return registry.experiments()
    except RegistryError:
        return []


def registered_bots(
    registry: "Registry | None", out: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    bots, skipped = [], []
    for version in registry.versions() if registry else []:
        packages = browser_packages(version.path) if version.browser_compatible else None
        if packages is None:
            skipped.append(version.ref)
            continue
        artifacts = visible_files(version.artifact_dir) if version.artifact_dir else []
        bots.append(install_bot(version.ref, [version.path, *artifacts], packages, out))
    return bots, skipped


def folder_bots(registered: set[str], out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    bots, skipped = [], []
    for bot_file in sorted(BOTS_DIR.glob("*/bot.py")):
        name = bot_file.parent.name
        if name in registered:
            continue
        packages = browser_packages(bot_file)
        if packages is None or not starts(bot_file):
            skipped.append(name)
            continue
        bots.append(install_bot(name, visible_files(bot_file.parent), packages, out))
    return bots, skipped


def model_entry(row: dict[str, Any], playable: set[str]) -> dict[str, Any]:
    return {
        **_log_row(row),
        "hypothesis": row["hypothesis"],
        "notes": row["notes"],
        "created_at": str(row["created_at"]),
        "playable": row["bot_ref"] in playable,
    }


def other_bots(
    registry: "Registry | None", stepped: set[str], playable: set[str]
) -> list[dict[str, Any]]:
    versions = registry.versions() if registry else []
    return [
        {"ref": version.ref, "kind": version.kind, "playable": version.ref in playable}
        for version in versions
        if version.name not in stepped
    ]


def win_rate_rows(registry: "Registry | None") -> list[dict[str, Any]]:
    rules = Rules(n_players=TEST_PLAYERS)
    return registry.win_rates(rules) if registry else []


def model_report(registry: "Registry | None", playable: set[str]) -> dict[str, Any]:
    rows = read_experiments(registry)
    stepped = {row["bot_ref"].partition("@")[0] for row in rows if row["bot_ref"]}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": [model_entry(row, playable) for row in rows],
        "others": other_bots(registry, stepped, playable),
        "win_rates": win_rate_rows(registry),
    }


def build_site() -> dict[str, Any]:
    shutil.rmtree(DIST, ignore_errors=True)
    ignored = shutil.ignore_patterns(DIST.name, "__pycache__", "README.md")
    shutil.copytree(WEB_DIR, DIST, ignore=ignored)

    wheel = build_wheel(DIST)
    registry = read_registry()
    versions, skipped_versions = registered_bots(registry, DIST)
    registered = {bot["name"].partition("@")[0] for bot in versions}
    registered.update(ref.partition("@")[0] for ref in skipped_versions)
    folders, skipped_folders = folder_bots(registered, DIST)
    bots = [*folders, *versions]
    report = model_report(registry, {bot["name"] for bot in bots})

    manifest = {"wheel": wheel.as_posix(), "bots": bots}
    (DIST / "bots.json").write_text(json.dumps(manifest, indent=2))
    (DIST / "models.json").write_text(json.dumps(report, indent=2))
    return {
        "dist": str(DIST),
        "bots": [bot["name"] for bot in bots],
        "skipped": skipped_versions + skipped_folders,
        "models": len(report["models"]),
    }


@app.command(help="Build the site, engine wheel, bots, and models into web/dist.")
def build(
    as_json: bool = typer.Option(False, "--json", help="Print the build summary as JSON."),
) -> None:
    summary = build_site()
    if as_json:
        typer.echo(json.dumps(summary))
        return
    typer.echo(f"Built {summary['dist']}")
    typer.echo(f"Bots: {', '.join(summary['bots']) or 'none'}")
    typer.echo(f"Models: {summary['models']}")
    if summary["skipped"]:
        typer.echo(f"Not browser compatible: {', '.join(summary['skipped'])}")


@app.command(help="Serve the built site locally.")
def serve(
    host: str = typer.Option("127.0.0.1", help="Address to bind."),
    port: int = typer.Option(8000, help="Port to listen on."),
    rebuild: bool = typer.Option(True, help="Rebuild before serving."),
) -> None:
    if rebuild or not DIST.exists():
        build_site()
    handler = partial(SimpleHTTPRequestHandler, directory=str(DIST))
    with ThreadingHTTPServer((host, port), handler) as server:
        typer.echo(f"Serving http://{host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            typer.echo("Stopped")
