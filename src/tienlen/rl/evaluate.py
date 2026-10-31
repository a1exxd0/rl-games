from collections.abc import Sequence
from dataclasses import asdict, dataclass

from tienlen.arena import wilson_interval
from tienlen.bots import TienLenBot
from tienlen.engine import Rules
from tienlen.rl.env import TienLenEnv, win_reward


@dataclass(frozen=True, slots=True)
class Evaluation:
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _play_out(env: TienLenEnv, bot: TienLenBot, seat: int, deal_seed: int) -> float:
    bot.reset(seat, env.rules, env.rng.getrandbits(63))
    step = env.reset(seat=seat, deal_seed=deal_seed)
    while not step.done:
        step = env.step(bot.act(step.view, step.legal))
    return step.reward


def evaluate(
    bot: TienLenBot,
    opponents: Sequence[TienLenBot],
    rules: Rules,
    *,
    deals: int,
    seed: int = 0,
) -> Evaluation:
    env = TienLenEnv(rules, opponents, reward=win_reward, seed=seed)
    wins = sum(
        _play_out(env, bot, seat, seed + deal)
        for deal in range(deals)
        for seat in range(rules.n_players)
    )
    games = deals * rules.n_players
    low, high = wilson_interval(int(wins), games)
    return Evaluation(games, int(wins), wins / games, low, high)
