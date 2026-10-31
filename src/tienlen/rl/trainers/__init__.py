from collections.abc import Callable
from importlib import import_module

TRAINERS = ("ppo",)


def get_trainer(name: str) -> Callable[..., str]:
    if name not in TRAINERS:
        raise ValueError(f"unknown trainer {name!r}, expected one of {TRAINERS}")
    return import_module(f"tienlen.rl.trainers.{name}").train
