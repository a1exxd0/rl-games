import contextlib
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from tienlen.arena import (
    Entrant,
    bot_seed,
    compute_ratings,
    head_to_head,
    is_rated,
    play_match,
    run_arena,
    schedule,
    validate_bot,
    wilson_interval,
)
from tienlen.bots import TienLenBot
from tienlen.bots.loader import load_bot
from tienlen.engine import PASS, Play, PlayerView, Rules, moves_to_list, replay
from tienlen.errors import BotLoadError

BOTS = Path(__file__).parent.parent / "bots"
GREEDY = BOTS / "greedy" / "bot.py"
RANDOM = BOTS / "random" / "bot.py"


class Crashes:
    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        pass

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        raise RuntimeError("boom")


class CrashesOnReset(Crashes):
    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        raise RuntimeError("boom")


class Illegal(Crashes):
    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        return Play.of((view.hand[0], view.hand[1])) if len(view.hand) > 2 else PASS


class Hangs(Crashes):
    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        while True:
            time.sleep(1)


class SwallowsErrors(Crashes):
    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        with contextlib.suppress(Exception):
            time.sleep(10)
        return legal[0]


class Recorder:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        self.seeds.append(seed)

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        return legal[0]


def greedy() -> TienLenBot:
    return load_bot(GREEDY)


def test_match_is_reproducible_and_replays():
    rules = Rules(n_players=2)
    first = play_match([greedy(), load_bot(RANDOM)], rules, seed=4)
    second = play_match([greedy(), load_bot(RANDOM)], rules, seed=4)
    assert first == second
    assert first.rules_id == rules.rules_id
    assert sorted(first.placements) == [0, 1]
    assert replay(rules, 4, first.moves).moves == first.moves
    assert moves_to_list(first.moves)


def test_match_rejects_wrong_seat_count():
    with pytest.raises(ValueError):
        play_match([greedy()], Rules(n_players=2), seed=1)


def test_match_uses_given_seat_names():
    result = play_match([greedy(), greedy()], Rules(n_players=2), 1, seats=["a@1", "b@1"])
    assert result.seats == ("a@1", "b@1")


def test_bots_never_get_the_deal_seed():
    bots = [Recorder(), Recorder()]
    play_match(bots, Rules(n_players=2), seed=12345)
    assert bots[0].seeds == [bot_seed(12345, 0)]
    assert bots[1].seeds == [bot_seed(12345, 1)]
    assert 12345 not in bots[0].seeds + bots[1].seeds
    assert bots[0].seeds != bots[1].seeds


@pytest.mark.parametrize("cheat", [Crashes, CrashesOnReset, Illegal])
@pytest.mark.parametrize("seat", [0, 1])
def test_a_broken_bot_forfeits_and_places_last(cheat, seat):
    bots = [greedy(), greedy()]
    bots[seat] = cheat()
    result = play_match(bots, Rules(n_players=2), seed=3)
    assert result.forfeits == (seat,)
    assert result.placements[-1] == seat
    assert replay(Rules(n_players=2), 3, result.moves).moves == result.moves


@pytest.mark.parametrize("slow", [Hangs, SwallowsErrors])
def test_a_slow_bot_forfeits_on_timeout(slow):
    started = time.monotonic()
    result = play_match([slow(), greedy()], Rules(n_players=2), seed=3, move_timeout=0.05)
    assert result.forfeits == (0,)
    assert time.monotonic() - started < 2


def test_forfeits_finish_four_seat_games():
    bots = [Crashes(), greedy(), Crashes(), greedy()]
    result = play_match(bots, Rules(), seed=8)
    assert set(result.forfeits) == {0, 2}
    assert set(result.placements[:2]) == {1, 3}
    assert result.placements[2:] == tuple(reversed(result.forfeits))


def test_schedule_rotates_seats_over_a_shared_deal():
    a, b = Entrant("a@1", GREEDY), Entrant("b@1", RANDOM)
    fixtures = schedule([a, b], 2, 4, seed=1)
    assert [fixture.seats for fixture in fixtures] == [(a, b), (b, a), (a, b), (b, a)]
    assert fixtures[0].seed == fixtures[1].seed
    assert fixtures[1].seed != fixtures[2].seed
    assert fixtures == schedule([a, b], 2, 4, seed=1)


def test_schedule_fills_tables_and_samples_combinations():
    a, b, c = (Entrant(f"{name}@1", GREEDY) for name in "abc")
    assert schedule([a, b], 4, 1, seed=1)[0].seats == (a, b, a, b)
    seen = {frozenset(fixture.seats) for fixture in schedule([a, b, c], 2, 6, seed=1)}
    assert seen == {frozenset(pair) for pair in ((a, b), (a, c), (b, c))}


def test_schedule_needs_entrants():
    with pytest.raises(ValueError):
        schedule([], 2, 1, seed=1)


def test_workers_match_inline_results():
    entrants = [Entrant("greedy@1", GREEDY), Entrant("random@1", RANDOM)]
    rules = Rules(n_players=2)
    inline = list(run_arena(entrants, rules, 8, seed=2))
    pooled = list(run_arena(entrants, rules, 8, seed=2, workers=2))
    assert inline == pooled


def test_ratings_reward_the_winner():
    games = [(["a@1", "b@1"], [0, 1])] * 20
    ratings = {rating.bot: rating for rating in compute_ratings(games)}
    assert ratings["a"].mu > ratings["b"].mu
    assert ratings["a"].ordinal > ratings["b"].ordinal
    assert ratings["a"].games == 20


def test_ratings_ignore_game_order():
    wins = [(["a@1", "b@1"], [0, 1])] * 8
    losses = [(["a@1", "b@1"], [1, 0])] * 4
    for games in (wins + losses, losses + wins):
        ratings = {rating.bot: rating for rating in compute_ratings(games)}
        assert ratings["a"].mu > ratings["b"].mu + 1
    assert compute_ratings(wins + losses) == compute_ratings(losses + wins)


def test_ratings_skip_unregistered_and_self_play():
    assert not is_rated(["Greedy", "Random"])
    assert not is_rated(["a@1", "a@1"])
    assert compute_ratings([(["a@1", "a@1"], [0, 1])]) == []


def test_ratings_merge_copies_of_a_ref():
    ratings = compute_ratings([(["a@1", "b@1", "a@1", "b@1"], [0, 2, 1, 3])])
    assert {rating.bot for rating in ratings} == {"a", "b"}
    assert all(rating.games == 1 for rating in ratings)


def test_head_to_head_counts_wins_and_ties():
    games = [
        (["a@1", "b@1"], [0, 1]),
        (["b@1", "a@1"], [1, 0]),
        (["a@1", "b@1"], [1, 0]),
        (["a@1", "b@1", "a@1", "b@1"], [0, 1, 3, 2]),
        (["c@1", "d@1"], [0, 1]),
    ]
    result = head_to_head(games, "a@1", "b@1")
    assert (result.wins, result.losses, result.ties) == (2, 1, 1)
    assert result.rate == pytest.approx(2 / 3)


def test_wilson_interval_brackets_the_rate():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.001)
    assert high == pytest.approx(0.596, abs=0.001)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_validate_passes_reference_bots():
    assert validate_bot(GREEDY).ok


def write_bot(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_validate_reports_load_failures(tmp_path):
    result = validate_bot(write_bot(tmp_path / "bot.py", "x = 1\n"))
    assert not result.ok
    assert "create_bot" in result.problems[0]


def test_validate_reports_forfeits(tmp_path):
    source = (
        "class Bad:\n"
        "    def reset(self, seat, rules, seed):\n"
        "        pass\n"
        "    def act(self, view, legal):\n"
        "        raise RuntimeError\n"
        "def create_bot(artifact_dir):\n"
        "    return Bad()\n"
    )
    result = validate_bot(write_bot(tmp_path / "bot.py", source))
    assert not result.ok
    assert "forfeited" in result.problems[0]


def test_load_bot_gets_artifact_dir(tmp_path):
    source = (
        "class Holds:\n"
        "    def __init__(self, path):\n"
        "        self.path = path\n"
        "    def reset(self, seat, rules, seed):\n"
        "        pass\n"
        "    def act(self, view, legal):\n"
        "        return legal[0]\n"
        "def create_bot(artifact_dir):\n"
        "    return Holds(artifact_dir)\n"
    )
    bot = load_bot(write_bot(tmp_path / "bot.py", source), tmp_path)
    assert vars(bot)["path"] == tmp_path


@pytest.mark.parametrize(
    "source",
    [
        "import nothing_here\n",
        "def create_bot(artifact_dir):\n    raise RuntimeError\n",
        "def create_bot(artifact_dir):\n    return 1\n",
    ],
)
def test_load_bot_wraps_failures(tmp_path, source):
    with pytest.raises(BotLoadError):
        load_bot(write_bot(tmp_path / "bot.py", source))
