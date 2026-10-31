# /// script
# dependencies = ["numpy"]
# ///
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SPEC = {"action_dim": 7, "action_hidden": [64]}
WEIGHTS_FILE = "weights.npz"
HAND_SIZE = 13
SUITS = 4
TWO_RANK = 12
EMPTY_STATE = np.zeros(0, dtype=np.float32)


def rank_of(card: int) -> int:
    return card // SUITS


def action_features(view, play) -> list[float]:
    left = len(view.hand) - len(play.cards)
    return [
        play.is_pass,
        len(play.cards) / HAND_SIZE,
        0.0 if play.is_pass else rank_of(play.top) / TWO_RANK,
        sum(rank_of(card) == TWO_RANK for card in play.cards) / SUITS,
        left / HAND_SIZE,
        left == 0,
        view.current is None,
    ]


def linear(weights: dict[str, np.ndarray], layer: str, inputs: np.ndarray) -> np.ndarray:
    return inputs @ weights[f"{layer}.weight"].T + weights[f"{layer}.bias"]


class LabBot:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.weights = weights

    def reset(self, seat: int, rules, seed: int) -> None:
        pass

    def encode(self, view, legal: Sequence) -> tuple[np.ndarray, np.ndarray]:
        actions = np.array([action_features(view, play) for play in legal], dtype=np.float32)
        return EMPTY_STATE, actions

    def scores(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        hidden = np.maximum(linear(self.weights, "action.0", actions), 0)
        return linear(self.weights, "output", hidden)[:, 0]

    def act(self, view, legal: Sequence):
        return legal[int(np.argmax(self.scores(*self.encode(view, legal))))]


def create_bot(artifact_dir: Path | None) -> LabBot:
    if artifact_dir is None:
        raise ValueError(f"lab bot needs an artifact_dir holding {WEIGHTS_FILE}")
    with np.load(Path(artifact_dir) / WEIGHTS_FILE) as archive:
        return LabBot({key: archive[key] for key in archive.files})
