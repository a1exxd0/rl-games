import os
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu

from tienlen.engine import Rules
from tienlen.rl.collect import Batch, CollectResult, CollectTask, Outcome, collect
from tienlen.rl.encoding import ACTION_DIM, STATE_DIM
from tienlen.rl.league import League, Opponent
from tienlen.rl.policy import WEIGHTS_FILE, Policy
from tienlen.rl.runs import RunDir

Mapper = Callable[[Callable, Iterable], Iterable]
NEURAL_BOT = Path("bots/neural/bot.py").resolve()
WIN_REWARD = 1.0
TASK_SCALE = 4


@dataclass(frozen=True, slots=True)
class PPOConfig:
    rules: dict[str, Any]
    iterations: int = 50
    episodes: int = 2000
    workers: int = 0
    seed: int = 0
    reward: str = "placement"
    snapshot_every: int = 10
    hidden: int = 256
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 3
    minibatch: int = 4096
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    self_share: float = 0.4
    snapshot_share: float = 0.3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})


class PolicyNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.state_1 = nn.Linear(STATE_DIM, hidden)
        self.state_2 = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, 1)
        self.action_1 = nn.Linear(hidden + ACTION_DIM, hidden)
        self.action_2 = nn.Linear(hidden, 1)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, segments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = relu(self.state_2(relu(self.state_1(states))))
        joined = torch.cat([hidden[segments], actions], dim=1)
        scores = self.action_2(relu(self.action_1(joined))).squeeze(-1)
        return scores, self.value(hidden).squeeze(-1)


def export_weights(model: PolicyNet) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}


def segment_log_softmax(
    scores: torch.Tensor, segments: torch.Tensor, n_segments: int
) -> torch.Tensor:
    peak = torch.full((n_segments,), -torch.inf).scatter_reduce(
        0, segments, scores.detach(), "amax"
    )
    shifted = scores - peak[segments]
    totals = torch.zeros(n_segments).index_add(0, segments, shifted.exp())
    return shifted - totals.log()[segments]


def minibatch_tensors(batch: Batch, offsets: np.ndarray, picks: np.ndarray) -> dict[str, Any]:
    counts = batch.action_counts[picks]
    starts = offsets[picks]
    flat = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, counts, strict=True)])
    first_rows = np.concatenate([[0], counts.cumsum()[:-1]])
    tensor = torch.from_numpy
    return {
        "states": tensor(batch.states[picks]),
        "actions": tensor(batch.actions[flat]),
        "segments": tensor(np.repeat(np.arange(len(picks)), counts)),
        "chosen": tensor(first_rows + batch.chosen[picks]),
        "log_probs": tensor(batch.log_probs[picks]),
        "returns": tensor(batch.returns[picks]),
        "advantages": tensor(batch.advantages[picks]),
    }


def _loss(model: PolicyNet, data: dict[str, Any], config: PPOConfig) -> tuple[torch.Tensor, dict]:
    n_steps = len(data["states"])
    scores, values = model(data["states"], data["actions"], data["segments"])
    log_probs = segment_log_softmax(scores, data["segments"], n_steps)

    gains = data["advantages"]
    gains = (gains - gains.mean()) / (gains.std() + 1e-8)
    ratio = (log_probs[data["chosen"]] - data["log_probs"]).exp()
    clipped = ratio.clamp(1 - config.clip, 1 + config.clip)
    policy_loss = -torch.min(ratio * gains, clipped * gains).mean()
    value_loss = (values - data["returns"]).pow(2).mean()
    entropy = -(log_probs.exp() * log_probs).sum() / n_steps

    total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    return total, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


def _update(
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    config: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    offsets = np.concatenate([[0], batch.action_counts.cumsum()[:-1]])
    logged: dict[str, list[float]] = defaultdict(list)
    for _ in range(config.epochs):
        order = rng.permutation(len(batch))
        for start in range(0, len(batch), config.minibatch):
            data = minibatch_tensors(batch, offsets, order[start : start + config.minibatch])
            loss, stats = _loss(model, data, config)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            for key, value in stats.items():
                logged[key].append(value)
    return {key: float(np.mean(values)) for key, values in logged.items()}


def _opponent_stats(outcomes: Iterable[Outcome]) -> dict[str, dict[str, float]]:
    rewards: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        for name in set(outcome.opponents):
            rewards[name].append(outcome.reward)
    return {
        name: {"games": len(seen), "win_rate": sum(r >= WIN_REWARD for r in seen) / len(seen)}
        for name, seen in sorted(rewards.items())
    }


def _tasks(
    league: League, model: PolicyNet, config: PPOConfig, rng: random.Random
) -> list[CollectTask]:
    n_opponents = Rules.from_dict(config.rules).n_players - 1
    weights = export_weights(model)
    seats = [tuple(league.sample(rng, n_opponents)) for _ in range(config.episodes)]
    n_tasks = max(1, config.workers * TASK_SCALE)
    return [
        CollectTask(
            weights,
            config.rules,
            config.reward,
            tuple(seats[shard::n_tasks]),
            rng.getrandbits(63),
            config.gae_lambda,
        )
        for shard in range(n_tasks)
    ]


@contextmanager
def worker_map(workers: int) -> Iterator[Mapper]:
    if workers == 0:
        yield map
        return
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    with get_context("spawn").Pool(workers) as pool:
        yield pool.imap_unordered


def _snapshot(run: RunDir, model: PolicyNet, iteration: int) -> Path:
    target = run.snapshot_dir(iteration)
    Policy(export_weights(model)).save(target / WEIGHTS_FILE)
    return target


def _save_checkpoint(
    run: RunDir, model: PolicyNet, optimizer: torch.optim.Optimizer, iteration: int
) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save({**state, "iteration": iteration}, run.checkpoint_path)


def _restore(run: RunDir, model: PolicyNet, optimizer: torch.optim.Optimizer) -> int:
    if not run.checkpoint_path.exists():
        return 0
    checkpoint = torch.load(run.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def _league(config: PPOConfig, opponents: Sequence[Opponent], run: RunDir) -> League:
    league = League(opponents, self_share=config.self_share, snapshot_share=config.snapshot_share)
    for snapshot in run.snapshots():
        league.add_snapshot(Opponent(f"snapshot-{snapshot.name}", NEURAL_BOT, snapshot))
    return league


def _iteration(
    run_map: Mapper,
    league: League,
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    rngs: tuple[random.Random, np.random.Generator],
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[CollectResult] = list(run_map(collect, _tasks(league, model, config, rngs[0])))
    collected = time.perf_counter()
    batch = Batch.concat(result.batch for result in results)
    outcomes = [outcome for result in results for outcome in result.outcomes]
    losses = _update(model, optimizer, batch, config, rngs[1])
    return {
        **losses,
        "steps": len(batch),
        "collect_seconds": collected - started,
        "update_seconds": time.perf_counter() - collected,
        "opponents": _opponent_stats(outcomes),
    }


def train(
    settings: dict[str, Any],
    run: RunDir,
    opponents: Sequence[Opponent],
    on_snapshot: Callable[[int, Path], None] = lambda iteration, path: None,
) -> str:
    config = PPOConfig.from_dict(settings)
    torch.manual_seed(config.seed)
    model = PolicyNet(config.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    start = _restore(run, model, optimizer)
    league = _league(config, opponents, run)
    rngs = (random.Random(config.seed + start), np.random.default_rng(config.seed + start))

    with worker_map(config.workers) as run_map:
        for iteration in range(start + 1, config.iterations + 1):
            if run.stop_requested():
                return "stopped"
            metrics = _iteration(run_map, league, model, optimizer, config, rngs)
            run.log_metrics({"iteration": iteration, **metrics})
            _save_checkpoint(run, model, optimizer, iteration)
            if iteration % config.snapshot_every == 0 or iteration == config.iterations:
                snapshot = _snapshot(run, model, iteration)
                league.add_snapshot(Opponent(f"snapshot-{snapshot.name}", NEURAL_BOT, snapshot))
                on_snapshot(iteration, snapshot)
    return "finished"
