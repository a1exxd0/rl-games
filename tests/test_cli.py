import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tienlen.cli import app

BOTS = Path(__file__).parent.parent / "bots"
GREEDY = str(BOTS / "greedy" / "bot.py")
RANDOM = str(BOTS / "random" / "bot.py")
runner = CliRunner()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TIENLEN_DATA", str(tmp_path / "data"))


def invoke(*args: str, code: int = 0) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == code, result.output
    return result.stdout


def as_json(*args: str):
    return json.loads(invoke(*args, "--json"))


@pytest.fixture
def seeded():
    invoke("bot", "register", GREEDY, "--no-test")
    invoke("bot", "register", RANDOM, "--no-test")
    return as_json(
        "arena", "run", "greedy", "random", "--games", "12", "--seed", "1", "--workers", "1"
    )


def test_full_flow(seeded):
    assert seeded["standings"][0]["ref"] == "greedy@1"
    assert sum(row["games"] for row in seeded["standings"]) == 24
    assert len(as_json("match", "list", "--arena", seeded["arena_id"])) == 12
    board = as_json("arena", "leaderboard", "--players", "2")
    assert [row["bot"] for row in board] == ["greedy", "random"]
    [h2h] = as_json("arena", "h2h", "greedy", "random")
    assert h2h["wins"] + h2h["losses"] + h2h["ties"] == 12


def test_arena_text_output(seeded):
    assert "arena " in invoke("arena", "run", "greedy", "random", "--games", "2", "--workers", "1")
    assert "greedy" in invoke("arena", "leaderboard")


def test_arena_defaults_cap_seats(seeded):
    result = as_json(
        "arena",
        "run",
        "greedy",
        "random",
        "greedy@1",
        "random@1",
        "greedy",
        "--games",
        "4",
        "--workers",
        "1",
    )
    assert result["rules_id"]


def test_arena_writes_in_batches(seeded):
    invoke(
        "arena", "run", "greedy", "random", "--games", "6", "--batch-size", "2", "--workers", "1"
    )
    assert len(as_json("match", "list", "--limit", "100")) == 18


def test_match_run_and_replay(seeded):
    [row] = as_json("match", "run", "greedy", "random", "--seed", "3")
    assert row["seats"] == ["greedy@1", "random@1"]
    assert as_json("match", "show", row["match_id"])["seats"] == row["seats"]
    assert as_json("match", "replay", row["match_id"])["seats"] == row["seats"]
    text = invoke("match", "replay", row["match_id"][:8])
    assert "seat 0 greedy@1:" in text
    assert "placements:" in text
    assert "greedy@1" in invoke("match", "list")
    assert "match_id" in invoke("match", "show", row["match_id"])


def test_match_run_checks_seats(seeded):
    invoke("match", "run", "greedy", "--players", "2", code=1)


def test_bot_commands(seeded, tmp_path):
    assert [row["bot"] for row in as_json("bot", "list")] == ["greedy", "random"]
    assert "greedy" in invoke("bot", "list")
    shown = as_json("bot", "show", "greedy")
    assert shown["version"] == 1
    assert shown["ratings"][0]["n_players"] == 2
    assert "code_sha" in invoke("bot", "show", "greedy")
    assert as_json("bot", "versions", "greedy")[0]["bot"] == "greedy"
    invoke("bot", "versions", "missing", code=1)
    assert as_json("bot", "retire", "random@1")["status"] == "retired"
    invoke("match", "run", "greedy", "random", code=1)
    out = tmp_path / "export"
    assert as_json("bot", "export", "greedy", "--out", str(out))["ref"] == "greedy@1"
    assert (out / "bot.py").read_text() == Path(GREEDY).read_text()


def test_bot_export_includes_artifacts(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "w.bin").write_bytes(b"x")
    invoke("bot", "register", GREEDY, "--artifact-dir", str(weights), "--kind", "nn", "--no-test")
    out = tmp_path / "export"
    invoke("bot", "export", "greedy", "--out", str(out))
    assert (out / "artifacts" / "w.bin").read_bytes() == b"x"
    assert as_json("bot", "show", "greedy")["artifact_dir"]


def test_bot_new_validates_and_registers(tmp_path):
    root = tmp_path / "mine"
    created = as_json("bot", "new", "my-bot", "--dir", str(root))
    assert "class MyBotBot" in Path(created["path"]).read_text()
    invoke("bot", "new", "my-bot", "--dir", str(root), code=1)
    report = as_json("bot", "validate", created["path"])
    assert report["ok"]
    assert report["browser_compatible"]
    assert as_json("bot", "register", created["path"], "--no-test")["ref"] == "my-bot@1"
    assert as_json("bot", "validate", "my-bot")["ok"]


def test_register_refuses_a_broken_bot(tmp_path):
    path = tmp_path / "broken" / "bot.py"
    path.parent.mkdir()
    path.write_text("x = 1\n")
    invoke("bot", "validate", str(path), code=1)
    invoke("bot", "register", str(path), code=1)
    invoke("bot", "register", str(tmp_path / "absent.py"), code=1)
    assert as_json("bot", "list") == []


def test_register_can_skip_the_check(tmp_path):
    path = tmp_path / "broken" / "bot.py"
    path.parent.mkdir()
    path.write_text("x = 1\n")
    assert as_json("bot", "register", str(path), "--no-check", "--no-test")["ref"] == "broken@1"


def test_db_commands(seeded):
    summary = {row["table"]: row["rows"] for row in as_json("db", "tables")}
    assert summary["matches"] == 12
    assert as_json("db", "init")[0]["table"] == "bots"
    rows = as_json("db", "query", "bots", "--where", "bot == 'greedy'", "--columns", "bot,kind")
    assert rows == [{"bot": "greedy", "kind": "rule"}]
    assert "greedy" in invoke("db", "query", "bots")
    invoke("db", "query", "bots", "--where", "bad !!", code=1)
    invoke("db", "query", "nope", code=1)
    assert as_json("db", "snapshots", "matches")[0]["operation"] == "append"
    assert as_json("db", "compact", "matches")[0]["files_after"] == 1
    assert len(as_json("db", "compact")) == 6


def test_dev_commands():
    rules = as_json("dev", "rules", "--players", "2", "--rule", "lock_on_pass=false")
    assert rules["n_players"] == 2
    assert rules["lock_on_pass"] is False
    assert as_json("dev", "bench", "--games", "2", "--players", "2")["games"] == 2


@pytest.mark.parametrize(
    "args",
    [
        ("dev", "rules", "--rule", "nonsense=1"),
        ("dev", "rules", "--rule", "no-equals"),
        ("dev", "rules", "--rule", "lock_on_pass=???"),
        ("dev", "rules", "--players", "9"),
    ],
)
def test_bad_rules_fail_cleanly(args):
    assert runner.invoke(app, list(args)).exit_code == 2


def test_unknown_bot_is_a_clean_error():
    result = runner.invoke(app, ["bot", "show", "nope"])
    assert result.exit_code == 1
    assert "unknown bot: nope" in result.stderr


@pytest.mark.parametrize("group", ["bot", "match", "arena", "train", "db", "web", "dev"])
def test_groups_are_wired(group):
    assert "Usage" in invoke(group, "--help")


def test_register_measures_win_rates_against_other_bots():
    invoke("bot", "register", GREEDY, "--no-test")
    report = as_json("bot", "register", RANDOM, "--test-games", "20", "--workers", "1")
    [row] = report["win_rates"]
    assert (row["first"], row["second"], row["games"]) == ("random@1", "greedy@1", 20)
    assert row["ci_low"] <= row["win_rate"] <= row["ci_high"]


def test_bot_test_plays_only_missing_games_and_keeps_no_replays(tmp_path):
    invoke("bot", "register", GREEDY, "--no-test")
    invoke("bot", "register", RANDOM, "--no-test")
    command = ("bot", "test", "greedy", "--workers", "1", "--seed", "3", "--games")
    assert as_json(*command, "10")[0]["games"] == 10
    assert as_json(*command, "10")[0]["games"] == 10
    assert as_json(*command, "14")[0]["games"] == 14
    assert not (tmp_path / "data" / "replays").exists()
    assert as_json("bot", "show", "greedy")["ratings"]


def test_bot_test_skips_retired_bots_and_covers_earlier_versions(tmp_path):
    invoke("bot", "register", GREEDY, "--no-test")
    invoke("bot", "register", RANDOM, "--no-test")
    changed = tmp_path / "greedy" / "bot.py"
    changed.parent.mkdir()
    changed.write_text(Path(GREEDY).read_text() + "\n\nVERSION = 2\n")
    invoke("bot", "register", str(changed), "--no-test")
    invoke("bot", "retire", "random@1")
    rows = as_json("bot", "test", "greedy@1", "--games", "4", "--workers", "1")
    assert [row["second"] for row in rows] == ["greedy@2"]


def test_bot_test_with_no_rivals_is_empty():
    invoke("bot", "register", GREEDY, "--no-test")
    assert as_json("bot", "test", "greedy", "--games", "4") == []
