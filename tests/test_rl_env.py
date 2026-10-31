import random
from pathlib import Path

import numpy as np
import pytest
from test_bots import load_bot

from tienlen.engine import PASS, Rules, legal_plays, new_game, view
from tienlen.rl.encoding import (
    ACTION_DIM,
    N_CARDS,
    STATE_DIM,
    encode_actions,
    encode_state,
)
from tienlen.rl.env import TienLenEnv, placement_reward, win_reward
from tienlen.rl.evaluate import evaluate
from tienlen.rl.league import CURRENT_POLICY, League, Opponent, from_refs


def random_env(n_players: int = 2, **kwargs) -> TienLenEnv:
    rules = Rules(n_players=n_players)
    return TienLenEnv(rules, [load_bot("random") for _ in range(n_players - 1)], seed=1, **kwargs)


def finish(env: TienLenEnv, rng: random.Random, **kwargs):
    step = env.reset(**kwargs)
    while not step.done:
        step = env.step(rng.choice(step.legal))
    return step


def test_placement_reward_spans_best_to_worst():
    assert [placement_reward(place, 4) for place in range(4)] == pytest.approx(
        [1, 1 / 3, -1 / 3, -1]
    )
    assert [placement_reward(place, 2) for place in range(2)] == [1, -1]


def test_win_reward_only_pays_first_place():
    assert [win_reward(place, 3) for place in range(3)] == [1, 0, 0]


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_env_finishes_from_every_seat(n_players):
    env = random_env(n_players)
    for seat in range(n_players):
        step = finish(env, random.Random(seat), seat=seat)
        assert step.done and step.legal == ()
        assert step.view.seat == seat


def test_env_gives_the_learner_its_turn_with_legal_plays():
    env = random_env()
    step = env.reset(seat=1, deal_seed=5)
    assert step.view.to_act == 1
    assert step.legal
    assert not step.done


def test_env_seat_is_random_by_default():
    env = random_env()
    assert {env.reset().view.seat for _ in range(30)} == {0, 1}


def test_env_is_deterministic_for_a_deal_and_seed():
    first, second = random_env(), random_env()
    assert first.reset(seat=0, deal_seed=3) == second.reset(seat=0, deal_seed=3)


def test_env_never_hands_bots_the_deal_seed():
    seeds = []

    class Spy:
        def reset(self, seat, rules, seed):
            seeds.append(seed)

        def act(self, view, legal):
            return legal[-1]

    TienLenEnv(Rules(n_players=2), [Spy()], seed=9).reset(seat=0, deal_seed=1234)
    assert seeds and 1234 not in seeds


def test_env_rejects_wrong_opponent_count():
    with pytest.raises(ValueError):
        TienLenEnv(Rules(n_players=4), [load_bot("random")]).reset()


def test_env_cannot_step_before_reset():
    with pytest.raises(RuntimeError):
        random_env().step(PASS)


def test_env_handles_instant_wins():
    rules = Rules(n_players=4, instant_wins=True)
    env = TienLenEnv(rules, [load_bot("random") for _ in range(3)], seed=0)
    assert finish(env, random.Random(0)).done


def test_encoding_shapes_and_types():
    state = new_game(Rules(n_players=3), seed=4)
    seen, legal = view(state, state.actor), legal_plays(state)
    assert encode_state(seen).shape == (STATE_DIM,)
    assert encode_actions(seen, legal).shape == (len(legal), ACTION_DIM)
    assert encode_state(seen).dtype == np.float32


def test_state_encoding_marks_own_hand():
    state = new_game(Rules(n_players=2), seed=4)
    seen = view(state, 0)
    assert np.flatnonzero(encode_state(seen)[:N_CARDS]).tolist() == list(seen.hand)


def test_action_encoding_marks_play_cards_and_leaves_pass_blank():
    env = random_env()
    step = env.reset(seat=0, deal_seed=8)
    while PASS not in step.legal:
        step = env.step(step.legal[0])
    encoded = encode_actions(step.view, step.legal)
    assert encoded[step.legal.index(PASS), :N_CARDS].sum() == 0
    assert np.flatnonzero(encoded[0, :N_CARDS]).tolist() == list(step.legal[0].cards)


def test_state_encoding_is_blind_to_hidden_cards():
    first = new_game(Rules(n_players=2), seed=4)
    seen = view(first, 0)
    resampled = view(seen.sample_world(random.Random(1)), 0)
    assert np.array_equal(encode_state(seen), encode_state(resampled))


def test_evaluate_rotates_every_seat():
    result = evaluate(load_bot("greedy"), [load_bot("random")], Rules(n_players=2), deals=50)
    assert result.games == 100
    assert 0.5 < result.win_rate <= 1
    assert result.ci_low <= result.win_rate <= result.ci_high


def test_league_prefers_self_when_empty():
    assert League([]).sample(random.Random(0), 5) == [CURRENT_POLICY] * 5


def test_league_samples_every_tier():
    fixed, snapshot = Opponent("fixed"), Opponent("snap")
    league = League([fixed])
    league.add_snapshot(snapshot)
    assert set(league.sample(random.Random(0), 200)) == {CURRENT_POLICY, fixed, snapshot}


def test_league_caps_snapshots_to_the_newest():
    league = League([], max_snapshots=2)
    for index in range(4):
        league.add_snapshot(Opponent(str(index)))
    assert [opponent.name for opponent in league.snapshots] == ["2", "3"]


def test_from_refs_keeps_ref_as_name():
    class Version:
        path = "bots/random/bot.py"
        artifact_dir = None

    assert from_refs(["random@latest"], lambda ref: Version) == [
        Opponent("random@latest", Path("bots/random/bot.py"), None)
    ]
