from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tienlen.bots import TienLenBot
from tienlen.bots.loader import load_bot
from tienlen.engine import HAND_SIZE, Play, PlayerView, Rules
from tienlen.errors import BotLoadError
from tienlen.lab.interface import LabBot
from tienlen.rl.collect import Batch
from tienlen.rl.env import Step, TienLenEnv, placement_reward
from tienlen.rl.league import CURRENT, Opponent

_loaded: dict[tuple[Opponent, int], TienLenBot] = {}


@dataclass(frozen=True, slots=True)
class RolloutTask:
    bot_path: Path
    weights_dir: Path
    rules: dict[str, Any]
    seats: tuple[tuple[Opponent, ...], ...]
    seed: int
    margin_coef: float


@dataclass(slots=True)
class RolloutResult:
    batch: Batch | None
    lengths: list[int]
    rewards: list[float]
    outcomes: list[tuple[str, float]]


class SampledBot:
    def __init__(self, policy: LabBot, seed: int) -> None:
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        state, actions = self.policy.encode(view, legal)
        return legal[_sample(self.policy.scores(state, actions), self.rng)[0]]


def log_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    return shifted - np.log(np.exp(shifted).sum())


def _sample(scores: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
    logits = log_softmax(scores)
    pick = int(np.argmax(logits + rng.gumbel(size=len(logits))))
    return pick, float(logits[pick])


def margin_bonus(step: Step, coef: float) -> float:
    counts = step.view.hand_counts
    own = counts[step.view.seat]
    return coef * ((sum(counts) - own) / (len(counts) - 1) - own) / HAND_SIZE


def _episode(
    env: TienLenEnv,
    policy: LabBot,
    rng: np.random.Generator,
    opponents: Sequence[TienLenBot],
    margin_coef: float,
) -> tuple[Batch | None, float]:
    states, actions, chosen, log_probs = [], [], [], []
    step = env.reset(opponents=opponents)
    while not step.done:
        state, encoded = policy.encode(step.view, step.legal)
        pick, log_prob = _sample(policy.scores(state, encoded), rng)
        states.append(state)
        actions.append(encoded)
        chosen.append(pick)
        log_probs.append(log_prob)
        step = env.step(step.legal[pick])

    reward = step.reward + margin_bonus(step, margin_coef)
    if not states:
        return None, reward
    size = len(states)
    return (
        Batch(
            np.stack(states),
            np.concatenate(actions),
            np.array([len(row) for row in actions], dtype=np.int64),
            np.array(chosen, dtype=np.int64),
            np.array(log_probs, dtype=np.float32),
            np.full(size, reward, dtype=np.float32),
            np.zeros(size, dtype=np.float32),
        ),
        reward,
    )


def _opponent_bot(opponent: Opponent, slot: int, policy: LabBot, seed: int) -> TienLenBot:
    if opponent.name == CURRENT:
        return SampledBot(policy, seed)
    key = (opponent, slot)
    if key not in _loaded:
        _loaded[key] = load_bot(opponent.bot_file, opponent.artifact_dir)
    return _loaded[key]


def _load_policy(task: RolloutTask) -> LabBot:
    bot = load_bot(task.bot_path, task.weights_dir)
    if not isinstance(bot, LabBot):
        raise BotLoadError(f"{task.bot_path}: create_bot must return a LabBot")
    return bot


def rollout(task: RolloutTask) -> RolloutResult:
    policy = _load_policy(task)
    rng = np.random.default_rng(task.seed)
    env = TienLenEnv(Rules.from_dict(task.rules), [], reward=placement_reward, seed=task.seed)
    batches, lengths, rewards, outcomes = [], [], [], []
    for seats in task.seats:
        bots = [_opponent_bot(o, slot, policy, task.seed) for slot, o in enumerate(seats)]
        batch, reward = _episode(env, policy, rng, bots, task.margin_coef)
        outcomes.extend((opponent.name, reward) for opponent in seats)
        if batch is not None:
            batches.append(batch)
            lengths.append(len(batch))
            rewards.append(reward)
    return RolloutResult(Batch.concat(batches) if batches else None, lengths, rewards, outcomes)
