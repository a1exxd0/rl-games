from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np

from tienlen.engine import Play, PlayerView, Rules
from tienlen.rl.encoding import encode_actions, encode_state

WEIGHTS_FILE = "weights.npz"
LAYERS = ("state_1", "state_2", "value", "action_1", "action_2")


def _linear(weights: Mapping[str, np.ndarray], layer: str, inputs: np.ndarray) -> np.ndarray:
    return inputs @ weights[f"{layer}.weight"].T + weights[f"{layer}.bias"]


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0)


class Policy:
    def __init__(self, weights: Mapping[str, np.ndarray]) -> None:
        self.weights = dict(weights)

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path) as archive:
            return cls({key: archive[key] for key in archive.files})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, allow_pickle=False, **self.weights)

    def forward(self, state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, float]:
        hidden = _relu(_linear(self.weights, "state_1", state))
        hidden = _relu(_linear(self.weights, "state_2", hidden))
        value = _linear(self.weights, "value", hidden)
        joined = np.concatenate([np.tile(hidden, (len(actions), 1)), actions], axis=1)
        scores = _linear(self.weights, "action_2", _relu(_linear(self.weights, "action_1", joined)))
        return scores[:, 0], float(value[0])

    def log_probs(self, state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, float]:
        scores, value = self.forward(state, actions)
        shifted = scores - scores.max()
        return shifted - np.log(np.exp(shifted).sum()), value


class PolicyBot:
    def __init__(self, policy: Policy, *, sample: bool = False) -> None:
        self.policy = policy
        self.sample = sample
        self.rng = np.random.default_rng()

    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        scores, _ = self.policy.forward(encode_state(view), encode_actions(view, legal))
        if self.sample:
            scores = scores + self.rng.gumbel(size=len(legal))
        return legal[int(np.argmax(scores))]
