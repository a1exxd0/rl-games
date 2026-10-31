import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tienlen.bots.loader import load_module
from tienlen.cli import app
from tienlen.engine import Rules, legal_plays, new_game, view
from tienlen.errors import RegistryError
from tienlen.lab.interface import WEIGHTS_FILE, LabBot, ModelSpec
from tienlen.lab.report import render
from tienlen.lab.step import (
    PROMOTED,
    REJECTED,
    StepRequest,
    current_best,
    is_promoted,
    merged_config,
    run_step,
    spread,
    step_number,
    summarize,
)
from tienlen.lab.suite import Pairing
from tienlen.registry import Registry

BOTS_DIR = Path(__file__).parent.parent / "bots"
STEP_BOTS = sorted(BOTS_DIR.glob("lab_*/bot.py"))
ALLOWED_IMPORTS = set(sys.stdlib_module_names) | {"numpy"}
step_ids = [path.parent.name for path in STEP_BOTS]


def imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.partition(".")[0])
    return found


def pairing(rate: float, low: float, high: float, opponent: str = "parent@1") -> Pairing:
    return Pairing(opponent, int(rate * 1000), 1000, rate, low, high, 0)


def experiment(registry: Registry, step: int, verdict: str, **fields) -> str:
    experiment_id = registry.start_experiment(
        step=step, parent_ref=None, change=f"change {step}", hypothesis="why", config={"lr": 1}
    )
    registry.update_experiment(experiment_id, verdict=verdict, **fields)
    return experiment_id


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("TIENLEN_DATA", str(tmp_path))
    opened = Registry.open(tmp_path)
    for name in ("random", "greedy"):
        opened.register(BOTS_DIR / name / "bot.py", kind="rule", rules=Rules(n_players=2))
    return opened


@pytest.mark.parametrize("path", STEP_BOTS, ids=step_ids)
def test_steps_are_frozen_against_shared_code(path):
    assert imports(path) <= ALLOWED_IMPORTS


@pytest.mark.parametrize("path", STEP_BOTS, ids=step_ids)
def test_step_inference_matches_the_training_network(path, tmp_path):
    torch = pytest.importorskip("torch")
    from tienlen.lab.net import LabNet, export_weights

    module = load_module(path)
    model = LabNet(ModelSpec(**module.SPEC))
    np.savez(tmp_path / WEIGHTS_FILE, **export_weights(model))
    bot = module.create_bot(tmp_path)
    assert isinstance(bot, LabBot)

    state = new_game(Rules(n_players=2), seed=3)
    legal = legal_plays(state)
    encoded_state, actions = bot.encode(view(state, state.actor), legal)
    assert bot.act(view(state, state.actor), legal) in legal
    scores, _ = model(
        torch.from_numpy(encoded_state)[None],
        torch.from_numpy(actions),
        torch.zeros(len(actions), dtype=torch.long),
    )
    assert bot.scores(encoded_state, actions) == pytest.approx(scores.detach().numpy(), abs=1e-4)


@pytest.mark.parametrize("path", STEP_BOTS, ids=step_ids)
def test_step_needs_its_weights(path):
    with pytest.raises(ValueError):
        load_module(path).create_bot(None)


def test_experiment_round_trip(registry):
    experiment_id = experiment(registry, 1, PROMOTED, bot_ref="a@1", notes="helped")
    row = registry.experiment(1)
    assert row["experiment_id"] == experiment_id
    assert (row["bot_ref"], row["verdict"], row["notes"]) == ("a@1", PROMOTED, "helped")
    assert json.loads(row["config"]) == {"lr": 1}


def test_experiment_hypothesis_is_stored_before_results(registry):
    registry.start_experiment(step=1, parent_ref=None, change="c", hypothesis="h", config={})
    row = registry.experiment(1)
    assert (row["hypothesis"], row["verdict"], row["results"]) == ("h", "pending", None)


def test_unknown_experiment_raises(registry):
    with pytest.raises(RegistryError):
        registry.experiment(9)
    with pytest.raises(RegistryError):
        registry.update_experiment("nope", notes="x")


def test_put_reference_replaces_the_previous_row(registry):
    registry.put_reference("reference", bot_ref="a@1", verdict="reference")
    registry.put_reference("reference", bot_ref="a@2", verdict="reference")
    assert [row["bot_ref"] for row in registry.experiments()] == ["a@2"]


def test_current_best_ignores_rejected_steps():
    rows = [
        {"step": 1, "verdict": PROMOTED},
        {"step": 2, "verdict": PROMOTED},
        {"step": 3, "verdict": REJECTED},
    ]
    best = current_best(rows)
    assert best is not None
    assert best["step"] == 2
    assert current_best([]) is None


def test_promotion_needs_the_interval_above_half():
    assert is_promoted(pairing(0.55, 0.52, 0.58))
    assert not is_promoted(pairing(0.52, 0.49, 0.55))
    assert not is_promoted(pairing(0.40, 0.37, 0.43))


def test_best_seed_is_chosen_without_looking_at_the_head_to_head():
    table = {
        "a@1": {
            "random": pairing(0.9, 0.88, 0.92),
            "greedy": pairing(0.7, 0.68, 0.72),
            "best": pairing(0.4, 0.38, 0.42),
        },
        "a@2": {
            "random": pairing(0.8, 0.78, 0.82),
            "greedy": pairing(0.6, 0.58, 0.62),
            "best": pairing(0.7, 0.68, 0.72),
        },
    }
    trained = [{"ref": "a@1", "seed": 1, "run_id": "x"}, {"ref": "a@2", "seed": 2, "run_id": "y"}]
    results = summarize(trained, table, 5.0)
    assert results["best"]["ref"] == "a@1"
    assert results["head_to_head"]["win_rate"] == 0.4
    assert results["summary"]["greedy"]["mean"] == pytest.approx(0.65)


def test_spread_handles_one_seed():
    assert spread([0.5]) == {"mean": 0.5, "stdev": 0.0, "min": 0.5, "max": 0.5}


@pytest.mark.parametrize(("folder", "number"), [("lab_01_linear", 1), ("lab_12_big_net", 12)])
def test_step_number_comes_from_the_folder(folder, number):
    assert step_number(folder) == number


def test_step_folder_must_match_the_pattern():
    with pytest.raises(RegistryError):
        step_number("neural")


def test_budget_cannot_change_silently():
    pytest.importorskip("torch")
    parent = {"config": json.dumps({"iterations": 60, "episodes": 4000})}
    with pytest.raises(RegistryError):
        merged_config(parent, {"iterations": 61}, budget_change=False)
    assert merged_config(parent, {"iterations": 61}, budget_change=True)["iterations"] == 61
    assert merged_config(parent, {"lr": 0.01}, budget_change=False)["iterations"] == 60


def test_step_inherits_the_parent_config():
    pytest.importorskip("torch")
    parent = {"config": json.dumps({"algo": "ppo", "epochs": 3})}
    assert merged_config(parent, {"lr": 0.01}, False)["algo"] == "ppo"


def test_unknown_settings_are_rejected():
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        merged_config(None, {"typo": 1}, False)


def tiny_request(folder: str, **overrides) -> StepRequest:
    settings = {"iterations": 1, "episodes": 8, "minibatch": 64, **overrides}
    return StepRequest(
        folder, "change", "hypothesis", settings, True, workers=0, seeds=(1, 2), games=20
    )


def test_run_step_records_seeds_and_a_verdict(registry):
    pytest.importorskip("torch")
    row = run_step(registry, tiny_request(step_ids[0]))
    results = json.loads(row["results"])
    assert row["verdict"] in (PROMOTED, REJECTED)
    assert len(row["train_run_ids"]) == 2
    assert [len(seed["pairings"]) for seed in results["seeds"]] == [3, 3]
    assert registry.resolve(row["bot_ref"]).browser_compatible


def test_run_step_supports_ppo_a_value_head_and_a_league(registry):
    pytest.importorskip("torch")
    first = run_step(registry, tiny_request(step_ids[0]))
    registry.update_experiment(first["experiment_id"], verdict=PROMOTED)
    second = StepRequest(
        step_ids[1],
        "change",
        "hypothesis",
        {
            "iterations": 2,
            "episodes": 8,
            "minibatch": 64,
            "algo": "ppo",
            "epochs": 2,
            "self_share": 0.5,
            "snapshot_share": 0.2,
            "snapshot_every": 1,
            "margin_coef": 0.1,
        },
        True,
        workers=0,
        seeds=(1,),
        games=10,
    )
    row = run_step(registry, second)
    assert row["parent_ref"] == first["bot_ref"]


def test_run_step_refuses_a_duplicate_step(registry):
    experiment(registry, 1, PROMOTED)
    with pytest.raises(RegistryError):
        run_step(registry, tiny_request("lab_01_linear"))


def results_json(greedy: float) -> str:
    entry = {"mean": greedy, "stdev": 0.01, "min": greedy, "max": greedy}
    return json.dumps(
        {
            "summary": {"random": entry, "greedy": entry},
            "head_to_head": pairing(0.6, 0.57, 0.63).to_dict(),
        }
    )


def test_cli_log_show_note_and_report(registry, tmp_path):
    experiment(registry, 1, PROMOTED, bot_ref="a@1", results=results_json(0.6))
    experiment(registry, 2, PROMOTED, bot_ref="b@1", results=results_json(0.7))
    experiment(registry, 3, REJECTED, bot_ref="c@1", results=results_json(0.65))
    registry.put_reference(
        "reference",
        bot_ref="neural@1",
        verdict="reference",
        results=json.dumps({"pairings": {"random": pairing(0.9, 0.88, 0.92, "r@1").to_dict()}}),
    )
    runner = CliRunner()

    logged = json.loads(runner.invoke(app, ["lab", "log", "--json"]).stdout)
    assert [row["verdict"] for row in logged][1:] == [PROMOTED, PROMOTED, REJECTED]
    assert json.loads(runner.invoke(app, ["lab", "show", "2", "--json"]).stdout)["step"] == 2
    assert runner.invoke(app, ["lab", "show", "2"]).exit_code == 0
    assert runner.invoke(app, ["lab", "note", "3", "it overfit"]).exit_code == 0

    output = tmp_path / "evolution.md"
    assert runner.invoke(app, ["lab", "report", "--output", str(output)]).exit_code == 0
    text = output.read_text()
    assert "| 2 | `b@1` | change 2 |" in text
    assert "it overfit" in text
    assert "+10.0% vs greedy" in text
    assert "## Outside reference" in text


def test_report_needs_experiments(registry):
    assert CliRunner().invoke(app, ["lab", "report"]).exit_code == 1


def test_render_shows_rating_when_known():
    row = {
        "step": 1,
        "bot_ref": "a@1",
        "change": "c",
        "hypothesis": "h",
        "results": results_json(0.5),
        "verdict": PROMOTED,
        "notes": None,
    }
    assert "| 12.5 |" in render([row], {"a@1": 12.5})
