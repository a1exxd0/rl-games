import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tienlen.cli import web
from tienlen.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def write_bot(folder: Path, source: str) -> Path:
    folder.mkdir(parents=True)
    path = folder / "bot.py"
    path.write_text(source)
    return path


@pytest.mark.parametrize(
    ("source", "packages"),
    [
        ("import random\nfrom tienlen.engine import PASS\n", []),
        ("from tienlen.bots import TienLenBot\n", []),
        ("import numpy as np\n", ["numpy"]),
        ("def f():\n    import numpy.linalg\n", ["numpy"]),
        ("import torch\n", None),
        ("from sklearn.linear_model import Ridge\n", None),
        ("from tienlen.registry import Registry\n", None),
        ("from tienlen import cli\n", None),
        ("from tienlen.rl.policy import Policy\n", ["numpy"]),
    ],
)
def test_browser_packages(tmp_path, source, packages):
    assert web.browser_packages(write_bot(tmp_path / "bot", source)) == packages


BOT_ENDING = "def create_bot(artifact_dir):\n    return object()\n"
NEEDS_WEIGHTS = (
    "def create_bot(artifact_dir):\n"
    "    if artifact_dir is None:\n"
    "        raise ValueError('needs weights')\n"
    "    return object()\n"
)


@pytest.fixture
def site(tmp_path, monkeypatch):
    bots = tmp_path / "bots"
    write_bot(bots / "plain", f"import random\n{BOT_ENDING}")
    numeric = write_bot(bots / "numeric", f"import numpy\n{NEEDS_WEIGHTS}")
    write_bot(bots / "no_weights", NEEDS_WEIGHTS)
    (numeric.parent / "weights.npz").write_bytes(b"x")
    (numeric.parent / "__pycache__").mkdir()
    write_bot(bots / "heavy", "import torch\n")

    monkeypatch.setattr(web, "BOTS_DIR", bots)
    monkeypatch.setattr(web, "DIST", tmp_path / "dist")
    monkeypatch.setenv("TIENLEN_DATA", str(tmp_path / "no-registry"))
    return tmp_path / "dist"


def test_build_writes_site_manifest_and_wheel(site):
    result = runner.invoke(web.app, ["build", "--json"])
    assert result.exit_code == 0, result.output

    summary = json.loads(result.output)
    assert summary["bots"] == ["numeric", "plain"]
    assert summary["skipped"] == ["heavy", "no_weights"]

    manifest = json.loads((site / "bots.json").read_text())
    assert manifest["bots"] == [
        {"name": "numeric", "files": ["bot.py", "weights.npz"], "packages": ["numpy"]},
        {"name": "plain", "files": ["bot.py"], "packages": []},
    ]
    assert (site / "bots" / "numeric" / "weights.npz").exists()
    assert not (site / "bots" / "heavy").exists()

    for file in ("index.html", "style.css", "js/worker.js", "py/session.py"):
        assert (site / file).exists()

    with zipfile.ZipFile(site / manifest["wheel"]) as wheel:
        assert "tienlen/engine/game.py" in wheel.namelist()


def test_build_replaces_stale_output(site):
    site.mkdir()
    (site / "stale.txt").write_text("old")
    assert runner.invoke(web.app, ["build"]).exit_code == 0
    assert not (site / "stale.txt").exists()


def test_build_reports_summary_as_text(site):
    result = runner.invoke(web.app, ["build"])
    assert "numeric, plain" in result.output
    assert "Not browser compatible: heavy, no_weights" in result.output


def test_build_without_a_registry_writes_empty_models(site):
    assert runner.invoke(web.app, ["build"]).exit_code == 0
    report = json.loads((site / "models.json").read_text())
    assert report["models"] == []
    assert report["others"] == []
    assert not (site.parent / "no-registry").exists()


def registered(tmp_path, name, source, weights=None):
    folder = tmp_path / "sources" / name
    bot_file = write_bot(folder, source)
    artifacts = None
    if weights is not None:
        artifacts = tmp_path / "artifacts" / name
        artifacts.mkdir(parents=True)
        (artifacts / "weights.npz").write_bytes(weights)
    return bot_file, artifacts


@pytest.fixture
def lab(site, tmp_path, monkeypatch):
    data = tmp_path / "lab"
    monkeypatch.setenv("TIENLEN_DATA", str(data))
    registry = Registry.open(data)

    bot_file, artifacts = registered(tmp_path, "step_one", f"import numpy\n{NEEDS_WEIGHTS}", b"w")
    registry.register(bot_file, name="step_one", kind="lab", artifact_dir=artifacts)
    bot_file, _ = registered(tmp_path, "gpu", f"import torch\n{BOT_ENDING}")
    registry.register(bot_file, name="gpu", kind="lab")
    bot_file, _ = registered(tmp_path, "rule", BOT_ENDING)
    registry.register(bot_file, name="rule", kind="rule")

    results = {
        "summary": {"random": {"mean": 0.9}, "greedy": {"mean": 0.6}},
        "head_to_head": {"win_rate": 0.55},
    }
    first = registry.start_experiment(
        step=1, parent_ref=None, change="first change", hypothesis="first guess", config={}
    )
    registry.update_experiment(
        first,
        bot_ref="step_one@1",
        results=json.dumps(results),
        verdict="promoted",
        notes="it helped",
    )
    second = registry.start_experiment(
        step=2, parent_ref="step_one@1", change="second change", hypothesis="second", config={}
    )
    registry.update_experiment(second, bot_ref="gpu@1", verdict="rejected")
    registry.start_experiment(
        step=3, parent_ref="step_one@1", change="third change", hypothesis="third", config={}
    )
    return site


def test_build_reads_steps_from_the_registry(lab):
    result = runner.invoke(web.app, ["build", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["models"] == 3

    report = json.loads((lab / "models.json").read_text())
    first, second, pending = report["models"]
    assert first["bot_ref"] == "step_one@1"
    assert first["verdict"] == "promoted"
    assert (first["vs_random"], first["vs_greedy"], first["h2h"]) == (0.9, 0.6, 0.55)
    assert first["hypothesis"] == "first guess"
    assert first["notes"] == "it helped"
    assert first["playable"] is True

    assert second["parent_ref"] == "step_one@1"
    assert second["playable"] is False

    assert pending["bot_ref"] is None
    assert pending["verdict"] == "pending"
    assert pending["playable"] is False

    assert [bot["ref"] for bot in report["others"]] == ["rule@1"]


def test_build_copies_registered_code_and_weights(lab):
    assert runner.invoke(web.app, ["build"]).exit_code == 0
    manifest = json.loads((lab / "bots.json").read_text())
    entries = {bot["name"]: bot for bot in manifest["bots"]}

    assert entries["step_one@1"]["files"] == ["bot.py", "weights.npz"]
    assert entries["step_one@1"]["packages"] == ["numpy"]
    assert (lab / "bots" / "step_one@1" / "weights.npz").read_bytes() == b"w"
    assert "gpu@1" not in entries
    assert not (lab / "bots" / "gpu@1").exists()


def test_build_does_not_write_to_the_registry(lab, tmp_path):
    data = tmp_path / "lab"
    before = {path: path.stat().st_mtime_ns for path in data.rglob("*") if path.is_file()}
    assert runner.invoke(web.app, ["build"]).exit_code == 0
    after = {path: path.stat().st_mtime_ns for path in data.rglob("*") if path.is_file()}
    assert after == before


def test_repo_bots_are_browser_compatible():
    for name in ("random", "greedy"):
        assert web.browser_packages(ROOT / "bots" / name / "bot.py") == []


def test_model_report_includes_recorded_win_rates(tmp_path):
    from tienlen.arena import MatchResult
    from tienlen.engine import Rules

    registry = Registry.open(tmp_path)
    rules = Rules(n_players=2)
    result = MatchResult(1, rules.rules_id, ("a@1", "b@1"), (0, 1), (), (), rules)
    registry.record_matches([result], replays=False)
    rows = web.model_report(registry, set())["win_rates"]
    assert {(row["first"], row["second"], row["win_rate"]) for row in rows} == {
        ("a@1", "b@1", 1.0),
        ("b@1", "a@1", 0.0),
    }
    assert registry.tables.read("matches")[0]["replay_uri"] is None
