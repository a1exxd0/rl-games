import importlib.util
from pathlib import Path

import pytest
from helpers import play, table

from tienlen.bots import TienLenBot
from tienlen.engine import PASS, Rules, apply, is_over, legal_plays, new_game, placements, view

BOTS_DIR = Path(__file__).parent.parent / "bots"


def load_bot(name: str) -> TienLenBot:
    spec = importlib.util.spec_from_file_location(f"bot_{name}", BOTS_DIR / name / "bot.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_bot(None)


@pytest.mark.parametrize("name", ["random", "greedy"])
def test_reference_bots_meet_the_protocol(name):
    assert isinstance(load_bot(name), TienLenBot)


@pytest.mark.parametrize("name", ["random", "greedy"])
@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_reference_bots_finish_games(name, n_players):
    rules = Rules(n_players=n_players)
    bots = [load_bot(name) for _ in range(n_players)]
    for seat, bot in enumerate(bots):
        bot.reset(seat, rules, seed=seat)
    state = new_game(rules, seed=11)
    while not is_over(state):
        seat = state.actor
        state = apply(state, bots[seat].act(view(state, seat), legal_plays(state)))
    assert sorted(placements(state)) == list(range(n_players))


def test_greedy_leads_its_lowest_card():
    state = table("5S 5C 9S", "3S")
    assert load_bot("greedy").act(view(state, 0), legal_plays(state)) == play("5S")


def test_greedy_beats_with_the_lowest_it_can():
    state = apply(table("3S 9S", "4S 8S 2H"), play("3S"))
    assert load_bot("greedy").act(view(state, 1), legal_plays(state)) == play("4S")


def test_greedy_passes_only_when_it_must():
    state = apply(table("2H 9S", "4S 8S"), play("2H"))
    assert load_bot("greedy").act(view(state, 1), legal_plays(state)) == PASS


def test_random_bot_is_seeded():
    state = new_game(Rules(first_lead_includes_lowest=False), seed=2)
    legal = legal_plays(state)
    first, second = load_bot("random"), load_bot("random")
    first.reset(0, state.rules, seed=7)
    second.reset(0, state.rules, seed=7)
    seen = view(state, state.actor)
    assert [first.act(seen, legal) for _ in range(20)] == [
        second.act(seen, legal) for _ in range(20)
    ]
