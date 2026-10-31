"""The contract a ladder step's bot file meets, on top of the TienLenBot one.

Each step is a self-contained `bots/lab_<nn>_<slug>/bot.py` that imports only
the standard library and numpy, so it keeps running after shared code changes.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from tienlen.bots import TienLenBot

WEIGHTS_FILE = "weights.npz"


@dataclass(frozen=True)
class ModelSpec:
    """The architecture a bot file declares as its module-level `SPEC` dict."""

    action_dim: int
    state_dim: int = 0
    state_hidden: list[int] = field(default_factory=list)
    action_hidden: list[int] = field(default_factory=list)
    value: bool = False
    deep_set: bool = False


@runtime_checkable
class LabBot(TienLenBot, Protocol):
    def encode(self, view, legal) -> tuple[np.ndarray, np.ndarray]:
        """Return the state vector and one row of action features per legal play."""
        ...

    def scores(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return one score per action row, using the bot's own numpy inference."""
        ...
