import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from tienlen.bots import TienLenBot
from tienlen.bots.loader import load_bot
from tienlen.engine import Rules
from tienlen.rl.encoding import ACTION_DIM, STATE_DIM, encode_actions, encode_state
from tienlen.rl.env import REWARDS, TienLenEnv
from tienlen.rl.league import CURRENT, Opponent
from tienlen.rl.policy import Policy, PolicyBot

_loaded_bots: dict[tuple[Opponent, int], TienLenBot] = {}


@dataclass(slots=True)
class Batch:
    states: np.ndarray
    actions: np.ndarray
    action_counts: np.ndarray
    chosen: np.ndarray
    log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray

    @classmethod
    def empty(cls) -> Self:
        floats = np.zeros(0, dtype=np.float32)
        ints = np.zeros(0, dtype=np.int64)
        return cls(
            np.zeros((0, STATE_DIM), dtype=np.float32),
            np.zeros((0, ACTION_DIM), dtype=np.float32),
            ints,
            ints,
            floats,
            floats,
            floats,
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        parts = list(batches)
        return cls(
            *(np.concatenate([getattr(part, name) for part in parts]) for name in cls.__slots__)
        )

    def __len__(self) -> int:
        return len(self.states)


@dataclass(frozen=True, slots=True)
class CollectTask:
    weights: Mapping[str, np.ndarray]
    rules: dict[str, Any]
    reward: str
    seats: tuple[tuple[Opponent, ...], ...]
    seed: int
    gae_lambda: float


@dataclass(slots=True)
class Outcome:
    opponents: tuple[str, ...]
    reward: float


@dataclass(slots=True)
class CollectResult:
    batch: Batch
    outcomes: list[Outcome] = field(default_factory=list)


def advantages(values: np.ndarray, reward: float, gae_lambda: float) -> np.ndarray:
    gains = np.zeros(len(values), dtype=np.float32)
    following_value, running = reward, 0.0
    for index in range(len(values) - 1, -1, -1):
        running = following_value - values[index] + gae_lambda * running
        gains[index] = running
        following_value = values[index]
    return gains


def play_episode(
    env: TienLenEnv,
    policy: Policy,
    rng: np.random.Generator,
    opponents: Sequence[TienLenBot],
    gae_lambda: float,
) -> tuple[Batch, float]:
    states, actions, chosen, log_probs, values = [], [], [], [], []
    step = env.reset(opponents=opponents)
    while not step.done:
        state = encode_state(step.view)
        encoded = encode_actions(step.view, step.legal)
        logits, value = policy.log_probs(state, encoded)
        pick = int(np.argmax(logits + rng.gumbel(size=len(logits))))
        states.append(state)
        actions.append(encoded)
        chosen.append(pick)
        log_probs.append(logits[pick])
        values.append(value)
        step = env.step(step.legal[pick])

    if not states:
        return Batch.empty(), step.reward
    predicted = np.array(values, dtype=np.float32)
    gains = advantages(predicted, step.reward, gae_lambda)
    return (
        Batch(
            np.stack(states),
            np.concatenate(actions),
            np.array([len(row) for row in actions], dtype=np.int64),
            np.array(chosen, dtype=np.int64),
            np.array(log_probs, dtype=np.float32),
            gains + predicted,
            gains,
        ),
        step.reward,
    )


def _load_opponent(opponent: Opponent, slot: int) -> TienLenBot:
    key = (opponent, slot)
    if key not in _loaded_bots:
        _loaded_bots[key] = load_bot(opponent.bot_file, opponent.artifact_dir)
    return _loaded_bots[key]


def _seat_bots(seats: Sequence[Opponent], policy: Policy) -> list[TienLenBot]:
    return [
        PolicyBot(policy, sample=True)
        if opponent.name == CURRENT
        else _load_opponent(opponent, slot)
        for slot, opponent in enumerate(seats)
    ]


def collect(task: CollectTask) -> CollectResult:
    policy = Policy(task.weights)
    rules = Rules.from_dict(task.rules)
    rng = np.random.default_rng(task.seed)
    env = TienLenEnv(
        rules,
        [],
        reward=REWARDS[task.reward],
        seed=random.Random(task.seed).getrandbits(63),
    )
    batches, outcomes = [Batch.empty()], []
    for seats in task.seats:
        bots = _seat_bots(seats, policy)
        batch, reward = play_episode(env, policy, rng, bots, task.gae_lambda)
        batches.append(batch)
        outcomes.append(Outcome(tuple(opponent.name for opponent in seats), reward))
    return CollectResult(Batch.concat(batches), outcomes)
