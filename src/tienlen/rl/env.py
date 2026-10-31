import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tienlen.bots import TienLenBot
from tienlen.engine import (
    GameState,
    Play,
    PlayerView,
    Rules,
    apply,
    is_over,
    legal_plays,
    new_game,
    placements,
    view,
)

Reward = Callable[[int, int], float]
SEED_BITS = 63


def placement_reward(place: int, n_players: int) -> float:
    return 1 - 2 * place / (n_players - 1)


def win_reward(place: int, n_players: int) -> float:
    return 1.0 if place == 0 else 0.0


REWARDS: dict[str, Reward] = {"placement": placement_reward, "win": win_reward}


@dataclass(frozen=True, slots=True)
class Step:
    view: PlayerView
    legal: tuple[Play, ...]
    reward: float
    done: bool


class TienLenEnv:
    def __init__(
        self,
        rules: Rules,
        opponents: Sequence[TienLenBot],
        *,
        reward: Reward = placement_reward,
        seed: int | None = None,
    ) -> None:
        self.rules = rules
        self.opponents = opponents
        self.reward = reward
        self.rng = random.Random(seed)
        self.seat = 0
        self._state: GameState | None = None
        self._bots: dict[int, TienLenBot] = {}

    def _seed(self) -> int:
        return self.rng.getrandbits(SEED_BITS)

    def _seat_opponents(self) -> dict[int, TienLenBot]:
        if len(self.opponents) != self.rules.n_players - 1:
            raise ValueError(f"expected {self.rules.n_players - 1} opponents")
        seats = [seat for seat in range(self.rules.n_players) if seat != self.seat]
        return dict(zip(seats, self.opponents, strict=True))

    def reset(
        self,
        *,
        seat: int | None = None,
        deal_seed: int | None = None,
        opponents: Sequence[TienLenBot] | None = None,
    ) -> Step:
        self.opponents = self.opponents if opponents is None else opponents
        self.seat = self.rng.randrange(self.rules.n_players) if seat is None else seat
        self._bots = self._seat_opponents()
        for bot_seat, bot in self._bots.items():
            bot.reset(bot_seat, self.rules, self._seed())
        return self._advance(new_game(self.rules, self._seed() if deal_seed is None else deal_seed))

    def step(self, play: Play) -> Step:
        if self._state is None or is_over(self._state):
            raise RuntimeError("reset the env before stepping")
        return self._advance(apply(self._state, play))

    def _advance(self, state: GameState) -> Step:
        while not is_over(state) and state.actor != self.seat:
            seat = state.actor
            state = apply(state, self._bots[seat].act(view(state, seat), legal_plays(state)))
        self._state = state
        return self._observe(state)

    def _observe(self, state: GameState) -> Step:
        seen = view(state, self.seat)
        if is_over(state):
            place = placements(state).index(self.seat)
            return Step(seen, (), self.reward(place, self.rules.n_players), True)
        return Step(seen, tuple(legal_plays(state)), 0.0, False)
