import json
import subprocess
import sys
from pathlib import Path

import pytest

from tienlen.arena import Entrant, MatchResult, play_match, run_arena
from tienlen.bots.loader import load_bot
from tienlen.engine import Rules, moves_from_list, replay
from tienlen.errors import RegistryError
from tienlen.registry import Registry, browser_compatible, declared_dependencies

BOTS = Path(__file__).parent.parent / "bots"
GREEDY = BOTS / "greedy" / "bot.py"
RANDOM = BOTS / "random" / "bot.py"
RULES = Rules(n_players=2)


@pytest.fixture
def registry(tmp_path) -> Registry:
    return Registry.open(tmp_path / "data")


def copy_bot(tmp_path: Path, name: str, extra: str = "") -> Path:
    path = tmp_path / name / "bot.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(GREEDY.read_text() + extra)
    return path


def arena(registry: Registry, games: int = 10, *refs: str) -> list[MatchResult]:
    entrants = [registry.resolve(ref).entrant for ref in refs or ("greedy", "random")]
    results = list(run_arena(entrants, RULES, games, seed=1))
    registry.record_matches(results, arena_id="a1")
    return results


def test_open_honours_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("TIENLEN_DATA", str(tmp_path / "elsewhere"))
    assert Registry.open().root == (tmp_path / "elsewhere").resolve()


def test_open_keeps_data(tmp_path):
    Registry.open(tmp_path).register(GREEDY, kind="rule")
    assert Registry.open(tmp_path).resolve("greedy").version == 1


def test_register_stores_an_immutable_copy(registry, tmp_path):
    path = copy_bot(tmp_path, "mine")
    first = registry.register(path, kind="rule")
    original = first.path.read_text()
    path.write_text(original + "\nVERSION = 2\n")
    second = registry.register(path, kind="rule")
    assert (first.ref, second.ref) == ("mine@1", "mine@2")
    assert first.path.read_text() == original
    assert second.path.read_text() != original
    assert first.path != second.path


def test_register_is_idempotent(registry):
    first = registry.register(GREEDY, kind="rule")
    assert registry.register(GREEDY, kind="rule") == first
    assert len(registry.versions("greedy")) == 1


def test_register_name_and_rules_create_versions(registry):
    named = registry.register(GREEDY, kind="rule", name="other")
    assert named.ref == "other@1"
    assert registry.register(GREEDY, kind="rule", name="other", rules=RULES).version == 2
    assert registry.resolve("other@2").rules_id == RULES.rules_id


def test_register_stores_artifacts_by_content(registry, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "w.npz").write_bytes(b"one")
    first = registry.register(GREEDY, kind="nn", artifact_dir=weights)
    (weights / "w.npz").write_bytes(b"two")
    second = registry.register(GREEDY, kind="nn", artifact_dir=weights)
    assert (first.artifact_dir / "w.npz").read_bytes() == b"one"
    assert (second.artifact_dir / "w.npz").read_bytes() == b"two"
    assert second.version == first.version + 1


def test_register_links_parent_and_run(registry):
    registry.register(RANDOM, kind="rule")
    run_id = registry.start_run("greedy", {"lr": 0.1})
    registry.register(GREEDY, kind="nn", parent="random", train_run_id=run_id)
    row = registry.version_rows("greedy")[0]
    assert row["parent_version"] == "random@1"
    assert row["train_run_id"] == run_id


@pytest.mark.parametrize("name", ["", "../x", "a b", "a@1"])
def test_register_rejects_bad_names(registry, name):
    with pytest.raises(RegistryError):
        registry.register(GREEDY, kind="rule", name=name or "!")


def test_register_rejects_missing_file(registry, tmp_path):
    with pytest.raises(RegistryError):
        registry.register(tmp_path / "bot.py", kind="rule")


def test_resolve_refs(registry, tmp_path):
    path = copy_bot(tmp_path, "mine")
    registry.register(path, kind="rule")
    path.write_text(path.read_text() + "\nX = 1\n")
    registry.register(path, kind="rule")
    assert registry.resolve("mine").version == 2
    assert registry.resolve("mine@latest").version == 2
    assert registry.resolve("mine@1").version == 1


@pytest.mark.parametrize("ref", ["missing", "greedy@9", "greedy@worst", "greedy@best"])
def test_resolve_rejects_unknown_refs(registry, ref):
    registry.register(GREEDY, kind="rule")
    with pytest.raises(RegistryError):
        registry.resolve(ref)


def test_retired_versions_stay_addressable(registry, tmp_path):
    path = copy_bot(tmp_path, "mine")
    registry.register(path, kind="rule")
    path.write_text(path.read_text() + "\nX = 1\n")
    registry.register(path, kind="rule")
    registry.retire("mine@2")
    assert registry.resolve("mine").version == 1
    assert registry.resolve("mine@2").version == 2
    registry.retire("mine@1")
    with pytest.raises(RegistryError):
        registry.resolve("mine")


def test_versions_lists_all_sorted(registry):
    registry.register(RANDOM, kind="rule")
    registry.register(GREEDY, kind="rule")
    assert [version.ref for version in registry.versions()] == ["greedy@1", "random@1"]
    assert [version.ref for version in registry.versions("random")] == ["random@1"]


def test_runs_lifecycle(registry):
    run_id = registry.start_run("greedy", {"lr": 0.1})
    assert registry.runs(run_id)[0]["status"] == "running"
    registry.finish_run(run_id, "completed", "runs/metrics.jsonl")
    run = registry.runs(run_id)[0]
    assert run["status"] == "completed"
    assert run["metrics_uri"] == "runs/metrics.jsonl"
    assert run["finished_at"] is not None
    assert json.loads(run["config"]) == {"lr": 0.1}
    assert len(registry.runs()) == 1


def test_finish_unknown_run_fails(registry):
    with pytest.raises(RegistryError):
        registry.finish_run("nope", "completed")


def test_record_matches_is_one_commit_per_batch(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    before = len(registry.tables.snapshots("matches"))
    arena(registry, 10)
    assert len(registry.tables.snapshots("matches")) == before + 1
    assert len(registry.matches()) == 10


def test_record_nothing_commits_nothing(registry):
    registry.record_matches([])
    assert registry.tables.snapshots("matches") == []


def test_matches_filters(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    arena(registry, 4)
    assert len(registry.matches(arena_id="a1")) == 4
    assert registry.matches(arena_id="other") == []
    assert len(registry.matches(bot="greedy")) == 4
    assert len(registry.matches(bot="greedy@1", limit=2)) == 2
    assert registry.matches(bot="absent") == []


def test_replays_are_stored_and_reproduce_the_game(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    results = arena(registry, 2)
    row = registry.matches()[0]
    stored = registry.replay(row["match_id"][:10])
    assert row["replay_uri"].startswith("replays/a1/")
    rules = Rules.from_dict(stored["rules"])
    state = replay(rules, stored["seed"], moves_from_list(stored["moves"]))
    assert stored["seed"] in {result.seed for result in results}
    assert state.moves == moves_from_list(stored["moves"])


def test_results_without_rules_have_no_replay(registry):
    result = play_match([load_bot(GREEDY), load_bot(RANDOM)], RULES, 1)
    bare = MatchResult(
        *(getattr(result, field) for field in MatchResult.__dataclass_fields__ if field != "rules")
    )
    [match_id] = registry.record_matches([bare])
    with pytest.raises(RegistryError):
        registry.replay(match_id)


@pytest.mark.parametrize("match_id", ["ffff", "not-hex'"])
def test_unknown_match_fails(registry, match_id):
    with pytest.raises(RegistryError):
        registry.match(match_id)


def test_ratings_update_and_rank(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    arena(registry, 40)
    registry.update_ratings(RULES.rules_id, 2)
    board = registry.leaderboard(RULES.rules_id, 2)
    assert [row["bot"] for row in board] == ["greedy", "random"]
    assert board[0]["games"] == 40
    assert registry.resolve("greedy@best").version == 1


def test_ratings_recompute_replaces_rows_and_keeps_history(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    arena(registry, 10)
    registry.update_ratings(RULES.rules_id, 2)
    arena(registry, 10)
    registry.update_ratings(RULES.rules_id, 2)
    board = registry.leaderboard()
    assert len(board) == 2
    assert board[0]["games"] == 20
    assert len(registry.tables.snapshots("ratings")) >= 2


def test_ratings_are_separate_per_rules(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    arena(registry, 4)
    entrants = [registry.resolve(ref).entrant for ref in ("greedy", "random")]
    registry.record_matches(list(run_arena(entrants, Rules(), 4, seed=2)))
    registry.update_ratings()
    assert {row["n_players"] for row in registry.leaderboard()} == {2, 4}
    assert {row["n_players"] for row in registry.leaderboard(n_players=4)} == {4}


def test_best_follows_ratings_and_skips_retired(registry, tmp_path):
    path = copy_bot(tmp_path, "mine")
    registry.register(path, kind="rule")
    registry.register(RANDOM, kind="rule")
    path.write_text(path.read_text() + "\nX = 1\n")
    registry.register(path, kind="rule")
    arena(registry, 40, "mine@1", "random")
    arena(registry, 2, "mine@2", "random")
    registry.update_ratings(RULES.rules_id, 2)
    assert registry.resolve("mine@best", rules=RULES).version == 1
    registry.retire("mine@1")
    assert registry.resolve("mine@best", rules=RULES).version == 2


def test_head_to_head_reports_each_rule_set(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    arena(registry, 40)
    [row] = registry.head_to_head("greedy", "random")
    assert row["wins"] + row["losses"] + row["ties"] == 40
    assert row["ci_low"] < row["win_rate"] < row["ci_high"]
    assert row["wins"] > row["losses"]


def test_tables_compact_and_migrate(registry):
    registry.register(GREEDY, kind="rule")
    registry.register(RANDOM, kind="rule")
    arena(registry, 2)
    arena(registry, 2)
    assert registry.tables.compact("matches") == {
        "table": "matches",
        "rows": 4,
        "files_before": 2,
        "files_after": 1,
    }
    assert len(registry.matches()) == 4
    registry.tables.migrate()
    assert registry.tables.compact("train_runs")["rows"] == 0


def test_read_rejects_bad_queries(registry):
    for where in ("bad !!", "missing == 1"):
        with pytest.raises(RegistryError):
            registry.tables.read("bots", where)
    with pytest.raises(RegistryError):
        registry.tables.read("nope")


def test_dependencies_read_from_inline_metadata(tmp_path):
    path = tmp_path / "bot.py"
    path.write_text('# /// script\n# dependencies = ["numpy>=2", "torch==2.1"]\n# ///\n')
    assert declared_dependencies(path.read_text()) == ["numpy>=2", "torch==2.1"]
    assert not browser_compatible(path)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", True),
        ("import random\nfrom tienlen.engine import PASS\n", True),
        ('# /// script\n# dependencies = ["numpy"]\n# ///\nimport numpy\n', True),
        ("import torch\n", False),
        ("from sklearn.tree import DecisionTreeClassifier\n", False),
        ('# /// script\n# dependencies = ["scipy"]\n# ///\n', False),
    ],
)
def test_browser_compatibility(tmp_path, source, expected):
    path = tmp_path / "bot.py"
    path.write_text(source)
    assert browser_compatible(path) is expected


def test_entrant_points_at_stored_code(registry):
    version = registry.register(GREEDY, kind="rule")
    assert version.entrant == Entrant("greedy@1", version.path, None)
    assert load_bot(version.path)


def test_concurrent_writers_keep_every_version(registry, tmp_path):
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tienlen.registry import Registry\n"
        "Registry.open(Path(sys.argv[1])).register(Path(sys.argv[2]), kind='rule', name='mine')\n"
    )
    paths = [copy_bot(tmp_path, f"copy{index}", f"\nX = {index}\n") for index in range(4)]
    workers = [
        subprocess.Popen([sys.executable, "-c", script, str(registry.root), str(path)])
        for path in paths
    ]
    assert [worker.wait() for worker in workers] == [0] * len(paths)
    assert sorted(version.version for version in registry.versions("mine")) == [1, 2, 3, 4]
