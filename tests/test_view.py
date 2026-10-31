import json
import random
from dataclasses import fields

from helpers import play, table

from tienlen.engine import (
    DECK,
    PASS,
    Move,
    PlayerView,
    Rules,
    apply,
    format_card,
    moves_from_list,
    moves_to_list,
    new_game,
    view,
)


def test_view_shows_public_state():
    state = table("3S 9S", "5S 6S", "7S 8S")
    state = apply(state, play("3S"))
    state = apply(state, PASS)
    seen = view(state, 2)
    assert seen.seat == 2
    assert seen.hand == state.hands[2]
    assert seen.hand_counts == (1, 2, 2)
    assert seen.current == play("3S")
    assert seen.last_player == 0
    assert seen.passed == frozenset({1})
    assert seen.to_act == 2
    assert seen.played == play("3S").cards
    assert seen.trick == (Move(0, play("3S")), Move(1, PASS))


def test_trick_resets_each_round_and_moves_keep_everything():
    state = table("3S 9S", "5S 6S")
    state = apply(state, play("9S"))
    state = apply(state, PASS)
    seen = view(state, 1)
    assert seen.trick == ()
    assert len(seen.moves) == 2


def test_view_has_no_hidden_fields():
    names = {field.name for field in fields(PlayerView)}
    assert not names & {"hands", "undealt", "seed"}


def test_view_dict_round_trips_through_json():
    state = apply(new_game(Rules(), seed=2), play("3S"))
    seen = view(state, 1)
    assert PlayerView.from_dict(json.loads(json.dumps(seen.to_dict()))) == seen


def test_view_dict_contains_only_known_cards():
    state = new_game(Rules(n_players=2), seed=5)
    seen = view(state, 0)
    known = {format_card(card) for card in seen.hand}
    hidden = {format_card(card) for card in DECK} - known
    tokens = json.dumps(seen.to_dict()).replace('"', " ").replace(",", " ").split()
    assert not hidden.intersection(tokens)


def test_move_log_round_trips_through_json():
    moves = (Move(0, play("3S 3C")), Move(1, PASS))
    assert moves_from_list(json.loads(json.dumps(moves_to_list(moves)))) == moves


def test_sample_world_keeps_my_hand_and_the_public_record():
    state = table(
        "3S 4S 9S", "5S 6S", "7S 8S", "10S JS", rules=Rules(first_lead_includes_lowest=False)
    )
    state = apply(state, play("9S"))
    seen = view(state, 1)
    world = seen.sample_world(random.Random(0))
    assert world.hands[1] == seen.hand
    assert tuple(len(hand) for hand in world.hands) == seen.hand_counts
    assert view(world, 1) == seen


def test_sample_world_leaves_real_state_untouched_and_varies():
    state = new_game(Rules(), seed=8)
    seen = view(state, 0)
    worlds = {seen.sample_world(random.Random(i)).hands for i in range(5)}
    assert len(worlds) > 1
