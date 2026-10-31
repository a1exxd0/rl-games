import random
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import nn

from tienlen.bots.loader import load_module
from tienlen.engine import Rules
from tienlen.lab.interface import WEIGHTS_FILE, ModelSpec
from tienlen.lab.net import LabNet, export_weights
from tienlen.lab.rollout import RolloutResult, RolloutTask, rollout
from tienlen.rl.collect import Batch, advantages
from tienlen.rl.league import League, Opponent
from tienlen.rl.runs import RunDir
from tienlen.rl.trainers.ppo import Mapper, minibatch_tensors, segment_log_softmax

BUDGET_KEYS = ("iterations", "episodes")
TASKS_PER_WORKER = 4


@dataclass(frozen=True)
class LabConfig:
    iterations: int = 60
    episodes: int = 4000
    algo: str = "reinforce"
    lr: float = 1e-3
    epochs: int = 1
    minibatch: int = 2048
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 1.0
    max_grad_norm: float = 1.0
    margin_coef: float = 0.0
    opponents: tuple[str, ...] = ("random@latest", "greedy@latest")
    self_share: float = 0.0
    snapshot_share: float = 0.0
    snapshot_every: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        names = {field.name for field in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown lab settings: {sorted(unknown)}")
        merged = {**data}
        if "opponents" in merged:
            merged["opponents"] = tuple(merged["opponents"])
        return cls(**merged)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "opponents": list(self.opponents)}


def model_spec(bot_path: Path) -> ModelSpec:
    return ModelSpec(**load_module(bot_path).SPEC)


def _save_weights(model: LabNet, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / WEIGHTS_FILE, **export_weights(model))
    return target


def _state_values(model: LabNet, states: np.ndarray) -> np.ndarray:
    if model.value is None:
        return np.zeros(len(states), dtype=np.float32)
    with torch.no_grad():
        hidden = torch.from_numpy(states)
        for layer in model.state:
            hidden = torch.relu(layer(hidden))
        return model.value(hidden).squeeze(-1).numpy()


def _fill_targets(result: RolloutResult, model: LabNet, gae_lambda: float) -> Batch:
    batch = result.batch
    if batch is None:
        raise ValueError("rollout recorded no moves")
    values = _state_values(model, batch.states)
    start = 0
    for length, reward in zip(result.lengths, result.rewards, strict=True):
        stop = start + length
        gains = advantages(values[start:stop], reward, gae_lambda)
        batch.advantages[start:stop] = gains
        batch.returns[start:stop] = gains + values[start:stop]
        start = stop
    return batch


def _policy_loss(picked: torch.Tensor, data: dict[str, Any], config: LabConfig) -> torch.Tensor:
    gains = data["advantages"]
    if config.algo == "reinforce":
        return -(picked * gains).mean()
    gains = (gains - gains.mean()) / (gains.std() + 1e-8)
    ratio = (picked - data["log_probs"]).exp()
    clipped = ratio.clamp(1 - config.clip, 1 + config.clip)
    return -torch.min(ratio * gains, clipped * gains).mean()


def _loss(model: LabNet, data: dict[str, Any], config: LabConfig) -> tuple[torch.Tensor, dict]:
    n_steps = len(data["states"])
    scores, values = model(data["states"], data["actions"], data["segments"])
    log_probs = segment_log_softmax(scores, data["segments"], n_steps)
    policy_loss = _policy_loss(log_probs[data["chosen"]], data, config)
    entropy = -(log_probs.exp() * log_probs).sum() / n_steps
    total = policy_loss - config.entropy_coef * entropy
    if values is not None:
        total = total + config.value_coef * (values - data["returns"]).pow(2).mean()
    return total, {"policy_loss": policy_loss.item(), "entropy": entropy.item()}


def _update(
    model: LabNet,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    config: LabConfig,
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


def _tasks(
    league: League,
    config: LabConfig,
    bot_path: Path,
    weights_dir: Path,
    rules: Rules,
    workers: int,
    rng: random.Random,
) -> list[RolloutTask]:
    seats = [tuple(league.sample(rng, rules.n_players - 1)) for _ in range(config.episodes)]
    n_tasks = max(1, workers * TASKS_PER_WORKER)
    return [
        RolloutTask(
            bot_path,
            weights_dir,
            rules.to_dict(),
            tuple(seats[shard::n_tasks]),
            rng.getrandbits(63),
            config.margin_coef,
        )
        for shard in range(n_tasks)
    ]


def _win_rates(outcomes: Iterable[tuple[str, float]]) -> dict[str, float]:
    rewards: dict[str, list[float]] = defaultdict(list)
    for name, reward in outcomes:
        rewards[name].append(reward)
    return {name: float(np.mean([r > 0 for r in seen])) for name, seen in sorted(rewards.items())}


@dataclass(frozen=True)
class SeedJob:
    bot_path: Path
    config: LabConfig
    rules: Rules
    seed: int
    run: RunDir
    fixed: Sequence[Opponent]
    workers: int


def _league(job: SeedJob) -> League:
    return League(
        job.fixed,
        self_share=job.config.self_share,
        snapshot_share=job.config.snapshot_share,
    )


def _iteration(
    job: SeedJob,
    run_map: Mapper,
    league: League,
    model: LabNet,
    optimizer: torch.optim.Optimizer,
    rngs: tuple[random.Random, np.random.Generator],
) -> dict[str, Any]:
    config = job.config
    weights_dir = _save_weights(model, job.run.path / "current")
    tasks = _tasks(league, config, job.bot_path, weights_dir, job.rules, job.workers, rngs[0])
    results = [result for result in run_map(rollout, tasks) if result.batch is not None]
    batch = Batch.concat(_fill_targets(result, model, config.gae_lambda) for result in results)
    stats = _update(model, optimizer, batch, config, rngs[1])
    outcomes = (outcome for result in results for outcome in result.outcomes)
    return {**stats, "steps": len(batch), "win_rates": _win_rates(outcomes)}


def train_seed(job: SeedJob, run_map: Mapper) -> Path:
    torch.manual_seed(job.seed)
    model = LabNet(model_spec(job.bot_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=job.config.lr)
    league = _league(job)
    rngs = (random.Random(job.seed), np.random.default_rng(job.seed))
    started = time.perf_counter()
    for iteration in range(1, job.config.iterations + 1):
        metrics = _iteration(job, run_map, league, model, optimizer, rngs)
        elapsed = time.perf_counter() - started
        job.run.log_metrics({"iteration": iteration, "seconds": elapsed, **metrics})
        if iteration % job.config.snapshot_every == 0:
            snapshot = _save_weights(model, job.run.snapshot_dir(iteration))
            league.add_snapshot(Opponent(f"snapshot-{iteration}", job.bot_path, snapshot))
    return _save_weights(model, job.run.path / "final")
