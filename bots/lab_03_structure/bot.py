# /// script
# dependencies = ["numpy"]
# ///
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SPEC = {"action_dim": 14}
WEIGHTS_FILE = "weights.npz"
HAND_SIZE = 13
SUITS = 4
TWO_RANK = 12
N_RANKS = 13
COMBOS = ("single", "pair", "triple", "quad", "sequence", "double_sequence")
EMPTY_STATE = np.zeros(0, dtype=np.float32)


def rank_of(card: int) -> int:
    return card // SUITS


def rank_counts(cards: Sequence[int]) -> list[int]:
    counts = [0] * N_RANKS
    for card in cards:
        counts[rank_of(card)] += 1
    return counts


def structure_features(view, play) -> list[float]:
    before = rank_counts(view.hand)
    after = list(before)
    for card in play.cards:
        after[rank_of(card)] -= 1
    broken = sum(0 < left < held for left, held in zip(after, before, strict=True))
    return [
        sum(count == 1 for count in after) / N_RANKS,
        broken / SUITS,
    ]


def action_features(view, play) -> list[float]:
    left = len(view.hand) - len(play.cards)
    combos = [str(play.type) == combo for combo in COMBOS[1:]]
    return [
        play.is_pass,
        len(play.cards) / HAND_SIZE,
        0.0 if play.is_pass else rank_of(play.top) / TWO_RANK,
        sum(rank_of(card) == TWO_RANK for card in play.cards) / SUITS,
        left / HAND_SIZE,
        left == 0,
        view.current is None,
        *structure_features(view, play),
        *combos,
    ]


class LabBot:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.weights = weights

    def reset(self, seat: int, rules, seed: int) -> None:
        pass

    def encode(self, view, legal: Sequence) -> tuple[np.ndarray, np.ndarray]:
        actions = np.array([action_features(view, play) for play in legal], dtype=np.float32)
        return EMPTY_STATE, actions

    def scores(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return actions @ self.weights["output.weight"][0] + self.weights["output.bias"][0]

    def act(self, view, legal: Sequence):
        return legal[int(np.argmax(self.scores(*self.encode(view, legal))))]


def create_bot(artifact_dir: Path | None) -> LabBot:
    if artifact_dir is None:
        raise ValueError(f"lab bot needs an artifact_dir holding {WEIGHTS_FILE}")
    with np.load(Path(artifact_dir) / WEIGHTS_FILE) as archive:
        return LabBot({key: archive[key] for key in archive.files})
