import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_bots import load_bot
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

import tienlen.cli.train as train_cli
from tienlen.bots.loader import load_bot as load_bot_file
from tienlen.engine import Rules
from tienlen.errors import BotLoadError
from tienlen.rl.collect import advantages
from tienlen.rl.encoding import ACTION_DIM, STATE_DIM
from tienlen.rl.league import Opponent
from tienlen.rl.policy import WEIGHTS_FILE, Policy, PolicyBot
from tienlen.rl.runs import RunDir
from tienlen.rl.trainers import get_trainer
from tienlen.rl.trainers.ppo import PolicyNet, export_weights, segment_log_softmax

BOTS_DIR = Path(__file__).parent.parent / "bots"


class FakeRegistry:
    def __init__(self):
        self.versions, self.finished = [], []

    def start_run(self, bot, config):
        return "run-1"

    def finish_run(self, run_id, status, metrics_uri=None):
        self.finished.append((run_id, status, metrics_uri))

    def resolve(self, ref):
        name = ref.split("@")[0]
        return SimpleNamespace(path=BOTS_DIR / name / "bot.py", artifact_dir=None)

    def register(self, path, *, kind, artifact_dir=None, parent=None, **details):
        self.versions.append(SimpleNamespace(parent=parent, artifact_dir=artifact_dir, **details))
        return SimpleNamespace(ref=f"neural@{len(self.versions)}")


RANDOM = Opponent("random", BOTS_DIR / "random" / "bot.py")


@pytest.fixture
def registry(monkeypatch, tmp_path):
    fake = FakeRegistry()
    monkeypatch.setenv("TIENLEN_DATA", str(tmp_path))
    monkeypatch.setattr(train_cli, "_open_registry", lambda: fake)
    return fake


def small_config(**overrides):
    return {
        "rules": Rules(n_players=2).to_dict(),
        "iterations": 2,
        "episodes": 8,
        "workers": 0,
        "hidden": 16,
        "snapshot_every": 1,
        "minibatch": 64,
        **overrides,
    }


def test_numpy_policy_matches_torch():
    model = PolicyNet(32)
    states, actions = torch.rand(1, STATE_DIM), torch.rand(7, ACTION_DIM)
    scores, value = model(states, actions, torch.zeros(7, dtype=torch.long))
    numpy_scores, numpy_value = Policy(export_weights(model)).forward(
        states[0].numpy(), actions.numpy()
    )
    assert numpy_scores == pytest.approx(scores.detach().numpy(), abs=1e-5)
    assert numpy_value == pytest.approx(value.item(), abs=1e-5)


def test_segment_log_softmax_matches_per_segment():
    scores = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.5])
    segments = torch.tensor([0, 0, 0, 1, 1])
    expected = torch.cat([scores[:3].log_softmax(0), scores[3:].log_softmax(0)])
    assert segment_log_softmax(scores, segments, 2).numpy() == pytest.approx(
        expected.numpy(), abs=1e-5
    )


def test_advantages_end_on_the_terminal_reward():
    gains = advantages(np.array([0.0, 0.5], dtype=np.float32), 1.0, 1.0)
    assert gains == pytest.approx([1.0, 0.5])


def test_policy_round_trips_through_npz(tmp_path):
    policy = Policy(export_weights(PolicyNet(8)))
    policy.save(tmp_path / WEIGHTS_FILE)
    loaded = Policy.load(tmp_path / WEIGHTS_FILE)
    assert loaded.weights.keys() == policy.weights.keys()


def test_policy_bot_picks_a_legal_play_in_a_full_game():
    rules = Rules(n_players=2)
    bot = PolicyBot(Policy(export_weights(PolicyNet(8))), sample=True)
    bot.reset(0, rules, 1)
    from tienlen.rl.env import TienLenEnv

    env = TienLenEnv(rules, [load_bot("random")], seed=0)
    step = env.reset(seat=0)
    while not step.done:
        step = env.step(bot.act(step.view, step.legal))


def test_neural_bot_file_loads_weights(tmp_path):
    Policy(export_weights(PolicyNet(8))).save(tmp_path / WEIGHTS_FILE)
    path = BOTS_DIR / "neural" / "bot.py"
    assert isinstance(load_bot_file(path, tmp_path), PolicyBot)
    with pytest.raises(BotLoadError):
        load_bot_file(path)


def test_trainer_logs_metrics_and_snapshots(tmp_path):
    run = RunDir(tmp_path / "run")
    run.create(small_config())
    seen = []
    status = get_trainer("ppo")(run.config(), run, [RANDOM], lambda i, path: seen.append(i))
    assert status == "finished"
    assert seen == [1, 2]
    assert [row["iteration"] for row in run.metrics()] == [1, 2]
    assert (run.snapshot_dir(2) / WEIGHTS_FILE).exists()


def test_trainer_resumes_from_its_checkpoint(tmp_path):
    run = RunDir(tmp_path / "run")
    run.create(small_config())
    train = get_trainer("ppo")
    train(run.config(), run, [RANDOM])
    train({**run.config(), "iterations": 3}, run, [RANDOM])
    assert [row["iteration"] for row in run.metrics()] == [1, 2, 3]


def test_trainer_honours_stop_requests(tmp_path):
    run = RunDir(tmp_path / "run")
    run.create(small_config())
    run.request_stop()
    assert get_trainer("ppo")(run.config(), run, [RANDOM]) == "stopped"
    assert run.metrics() == []


def test_unknown_trainer_is_rejected():
    with pytest.raises(ValueError):
        get_trainer("nope")


def start_args():
    return [
        "start",
        "--opponents",
        "random@latest",
        "--iterations",
        "2",
        "--episodes",
        "8",
        "--workers",
        "0",
        "--snapshot-every",
        "1",
        "--set",
        "hidden=16",
        "--set",
        "minibatch=64",
        "--json",
    ]


def test_cli_start_registers_snapshots_with_a_parent_chain(registry):
    result = CliRunner().invoke(train_cli.app, start_args())
    assert result.exit_code == 0, result.output
    assert [version.parent for version in registry.versions] == [None, "neural@1"]
    assert registry.finished[0][:2] == ("run-1", "finished")


def test_cli_list_show_stop_and_resume(registry):
    runner = CliRunner()
    runner.invoke(train_cli.app, start_args())

    listed = runner.invoke(train_cli.app, ["list", "--json"], catch_exceptions=False)
    assert json.loads(listed.stdout)[0]["run_id"] == "run-1"
    shown = runner.invoke(train_cli.app, ["show", "run-1", "--json"])
    assert json.loads(shown.stdout)["iteration"] == 2

    assert runner.invoke(train_cli.app, ["stop", "run-1"]).exit_code == 0
    resumed = runner.invoke(train_cli.app, ["resume", "run-1", "--iterations", "3", "--json"])
    assert json.loads(resumed.stdout)["iteration"] == 3
    assert registry.versions[-1].parent == "neural@2"


def test_cli_list_includes_runs_without_a_trainer(registry, tmp_path):
    run = RunDir(tmp_path / "runs" / "lab-run")
    run.create({"bot": "lab_01_linear", "iterations": 60})
    run.log_metrics({"iteration": 1, "win_rates": {}})
    listed = CliRunner().invoke(train_cli.app, ["list", "--json"], catch_exceptions=False)
    assert json.loads(listed.stdout)[0]["trainer"] == "lab"


def test_cli_show_reports_missing_runs(registry):
    assert CliRunner().invoke(train_cli.app, ["show", "ghost"]).exit_code == 1
