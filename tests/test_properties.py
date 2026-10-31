import json
import random
from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from tienlen.engine import (
    DECK,
    GameState,
    Move,
    PlayerView,
    Rules,
    apply,
    format_card,
    is_legal,
    is_over,
    legal_plays,
    moves_from_list,
    moves_to_list,
    new_game,
    placements,
    replay,
    view,
)

MAX_MOVES = 52 * 4 + 52

rules_strategy = st.builds(
    Rules,
    n_players=st.integers(2, 4),
    lock_on_pass=st.booleans(),
    chop_after_pass=st.booleans(),
    instant_wins=st.booleans(),
    first_lead_includes_lowest=st.booleans(),
    forbid_finish_on_two=st.booleans(),
)
seeds = st.integers(0, 2**32)


def random_states(rules: Rules, seed: int, chooser: int):
    rng = random.Random(chooser)
    state = new_game(rules, seed)
    yield state
    while not is_over(state):
        state = apply(state, rng.choice(legal_plays(state)))
        yield state


def all_cards(state: GameState) -> list[int]:
    held = [card for hand in state.hands for card in hand]
    played = [card for move in state.moves for card in move.play.cards]
    return sorted([*held, *played, *state.undealt])


@settings(max_examples=150, deadline=None)
@given(rules_strategy, seeds, seeds)
def test_games_terminate_and_conserve_cards(rules, seed, chooser):
    moves = 0
    for state in random_states(rules, seed, chooser):
        moves = len(state.moves)
        assert moves <= MAX_MOVES
        assert all_cards(state) == list(DECK)
    assert is_over(state)
    assert sorted(placements(state)) == list(range(rules.n_players))
    assert state.moves == () or sum(1 for hand in state.hands if hand) <= 1


@settings(max_examples=100, deadline=None)
@given(rules_strategy, seeds, seeds)
def test_every_legal_play_is_accepted_and_someone_can_always_act(rules, seed, chooser):
    for state in random_states(rules, seed, chooser):
        if is_over(state):
            break
        plays = legal_plays(state)
        assert plays
        assert len(set(plays)) == len(plays)
        for play in plays:
            assert is_legal(state, play)
            after = apply(state, play)
            assert after.moves[-1] == Move(state.to_act, play)


@settings(max_examples=100, deadline=None)
@given(rules_strategy, seeds, seeds)
def test_view_leaks_no_hidden_cards(rules, seed, chooser):
    for state in random_states(rules, seed, chooser):
        for seat in range(rules.n_players):
            seen = view(state, seat)
            known = set(seen.hand) | set(seen.played)
            hidden = {format_card(card) for card in DECK} - {format_card(c) for c in known}
            tokens = json.dumps(seen.to_dict()).replace('"', " ").replace(",", " ")
            assert not hidden.intersection(tokens.split())


@settings(max_examples=100, deadline=None)
@given(rules_strategy, seeds, seeds)
def test_view_round_trips_all_game_long(rules, seed, chooser):
    for state in random_states(rules, seed, chooser):
        seen = view(state, state.first_leader)
        assert PlayerView.from_dict(json.loads(json.dumps(seen.to_dict()))) == seen


@settings(max_examples=100, deadline=None)
@given(rules_strategy, seeds, seeds)
def test_replay_is_deterministic(rules, seed, chooser):
    *_, final = random_states(rules, seed, chooser)
    logged = moves_from_list(json.loads(json.dumps(moves_to_list(final.moves))))
    assert replay(rules, seed, logged) == final


@settings(max_examples=100, deadline=None)
@given(rules_strategy, seeds, seeds, seeds)
def test_sampled_worlds_are_consistent(rules, seed, chooser, sampler):
    rng = random.Random(sampler)
    for state in random_states(rules, seed, chooser):
        if is_over(state):
            break
        seat = rng.randrange(rules.n_players)
        seen = view(state, seat)
        world = seen.sample_world(rng)

        assert view(world, seat) == seen
        assert all_cards(world) == list(DECK)

        played_by = Counter()
        originals = [set(hand) for hand in world.hands]
        for move in world.moves:
            originals[move.seat].update(move.play.cards)
            played_by[move.seat] += len(move.play.cards)
        lowest = min(card for cards in originals for card in cards)
        assert lowest in originals[world.first_leader]
        if world.moves and rules.first_lead_includes_lowest:
            assert lowest in world.moves[0].play.cards

        plays = legal_plays(world)
        assert plays
        apply(world, rng.choice(plays))
