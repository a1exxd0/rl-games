import pytest
from helpers import play, table

from tienlen.engine import (
    DECK,
    HAND_SIZE,
    PASS,
    IllegalPlay,
    Move,
    Play,
    Rules,
    apply,
    instant_win,
    is_legal,
    is_over,
    legal_plays,
    new_game,
    parse_card,
    parse_cards,
    placements,
    replay,
)


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_deal_gives_thirteen_each_and_keeps_the_rest_undealt(n_players):
    state = new_game(Rules(n_players=n_players), seed=3)
    assert [len(hand) for hand in state.hands] == [HAND_SIZE] * n_players
    assert len(state.undealt) == len(DECK) - HAND_SIZE * n_players
    dealt = [card for hand in state.hands for card in hand] + list(state.undealt)
    assert sorted(dealt) == list(DECK)


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_first_leader_holds_the_lowest_dealt_card(n_players):
    for seed in range(50):
        state = new_game(Rules(n_players=n_players), seed)
        lowest = min(card for hand in state.hands for card in hand)
        assert lowest in state.hands[state.to_act]
        assert state.first_leader == state.to_act


def test_same_seed_same_deal():
    assert new_game(Rules(), 9) == new_game(Rules(), 9)
    assert new_game(Rules(), 9).hands != new_game(Rules(), 10).hands


def test_first_lead_must_include_the_lowest_card():
    state = new_game(Rules(), seed=1)
    lowest = state.hands[state.to_act][0]
    plays = legal_plays(state)
    assert plays
    assert all(lowest in p.cards for p in plays)
    assert PASS not in plays


def test_first_lead_toggle_frees_the_opening():
    state = new_game(Rules(first_lead_includes_lowest=False), seed=1)
    lowest = state.hands[state.to_act][0]
    assert any(lowest not in p.cards for p in legal_plays(state))


def test_leader_cannot_pass():
    state = table("3S 4S", "5S 6S")
    assert PASS not in legal_plays(state)
    assert not is_legal(state, PASS)


def test_responder_may_pass_and_must_beat_the_current_play():
    state = apply(table("3S 9S", "5S 6S"), play("9S"))
    assert legal_plays(state) == [PASS]
    state = apply(table("3S 4S", "5S 6S"), play("4S"))
    assert legal_plays(state) == [play("5S"), play("6S"), PASS]


def test_turn_order_goes_around_the_table():
    state = table("3S 9S", "4S 9C", "5S 9D", "6S 9H")
    state = apply(state, play("3S"))
    assert state.to_act == 1
    state = apply(state, play("4S"))
    assert state.to_act == 2
    state = apply(state, play("5S"))
    assert state.to_act == 3


def test_round_ends_when_everyone_else_passes():
    state = table("3S 9S", "4S 9C", "5S 9D")
    state = apply(state, play("9S"))
    state = apply(state, PASS)
    state = apply(state, PASS)
    assert state.to_act == 0
    assert state.current is None
    assert state.passed == frozenset()
    assert state.round_start == 3


def test_last_player_leads_the_next_round():
    state = table("3S 9S", "4S 9C", "5S 9D")
    state = apply(state, play("3S"))
    state = apply(state, play("9C"))
    state = apply(state, PASS)
    state = apply(state, PASS)
    assert state.to_act == 1
    assert state.current is None


def test_a_passed_seat_is_skipped_until_the_round_ends():
    state = table("3S 5S 9S", "4S 6S 9C", "7S 8S 9D")
    state = apply(state, play("3S"))
    state = apply(state, PASS)
    state = apply(state, play("7S"))
    state = apply(state, play("9S"))
    assert state.passed == frozenset({1})
    assert state.to_act == 2
    state = apply(state, PASS)
    assert state.to_act == 0
    assert state.current is None


def test_unlocked_pass_lets_a_seat_back_in():
    rules = Rules(n_players=3, lock_on_pass=False, first_lead_includes_lowest=False)
    state = table("3S 5S 9S", "4S 6S 9C", "7S 8S 9D", rules=rules)
    state = apply(state, play("3S"))
    state = apply(state, PASS)
    state = apply(state, play("7S"))
    state = apply(state, play("9S"))
    assert state.to_act == 1
    assert state.passed == frozenset()
    assert play("9C") in legal_plays(state)


def test_finished_player_gets_placed_and_skipped():
    state = table("3S", "4S 9C", "5S 9D")
    state = apply(state, play("3S"))
    assert placements(state) == (0,)
    assert state.to_act == 1
    state = apply(state, play("4S"))
    assert state.to_act == 2
    state = apply(state, play("5S"))
    assert state.to_act == 1


def test_lead_passes_to_next_holder_when_the_last_player_went_out():
    state = table("9S", "4S 9C", "5S 9D")
    state = apply(state, play("9S"))
    state = apply(state, PASS)
    state = apply(state, PASS)
    assert state.to_act == 1
    assert state.current is None
    assert state.last_player is None


def test_lead_skips_empty_seats_when_passing_on():
    state = table("3S 9S", "", "4S 8S", "5S 7S")
    state = apply(state, play("9S"))
    assert state.to_act == 2
    assert state.hands[0] == parse_cards("3S")


def test_game_ends_when_one_player_holds_cards():
    state = table("3S", "4S", "5S 9D")
    state = apply(state, play("3S"))
    assert not is_over(state)
    state = apply(state, play("4S"))
    assert is_over(state)
    assert placements(state) == (0, 1, 2)
    assert legal_plays(state) == []
    assert not is_legal(state, PASS)


def test_two_player_game_ends_on_the_first_finish():
    state = apply(table("3S", "4S 5S"), play("3S"))
    assert is_over(state)
    assert placements(state) == (0, 1)


def test_apply_rejects_cards_not_in_hand():
    with pytest.raises(IllegalPlay):
        apply(table("3S 4S", "5S"), play("5S"))


def test_apply_rejects_a_play_that_does_not_beat_current():
    state = apply(table("3S 9S", "5S"), play("9S"))
    with pytest.raises(IllegalPlay):
        apply(state, play("5S"))


def test_apply_rejects_a_mislabelled_play():
    forged = Play(parse_cards("3S 4S"), play("3S 3C").type)
    assert not is_legal(table("3S 4S", "5S"), forged)
    with pytest.raises(IllegalPlay):
        apply(table("3S 4S", "5S"), forged)


def test_chop_is_legal_in_play():
    state = apply(table("2S 9S", "3S 3C 3D 3H 9C"), play("2S"))
    assert play("3S 3C 3D 3H") in legal_plays(state)


def test_locked_seat_cannot_chop_by_default():
    state = table("4S 2S 9S", "3S 3C 3D 3H 9C", "5S 6S 7S")
    state = apply(state, play("4S"))
    state = apply(state, PASS)
    state = apply(state, play("7S"))
    state = apply(state, play("2S"))
    assert state.to_act == 2


def test_chop_after_pass_lets_a_locked_seat_chop_only():
    rules = Rules(n_players=3, chop_after_pass=True, first_lead_includes_lowest=False)
    state = table("4S 2S 9S", "3S 3C 3D 3H 9C", "5S 6S 7S", rules=rules)
    state = apply(state, play("4S"))
    state = apply(state, PASS)
    state = apply(state, play("7S"))
    state = apply(state, play("2S"))
    assert state.to_act == 1
    assert legal_plays(state) == [play("3S 3C 3D 3H"), PASS]
    assert not is_legal(state, play("9C"))

    chopped = apply(state, play("3S 3C 3D 3H"))
    assert 1 not in chopped.passed
    assert chopped.last_player == 1


def test_locked_seat_without_a_chop_stays_skipped():
    rules = Rules(n_players=3, chop_after_pass=True, first_lead_includes_lowest=False)
    state = table("4S 2S 9S", "3S 8C 9C", "5S 6S 7S", rules=rules)
    state = apply(state, play("4S"))
    state = apply(state, PASS)
    state = apply(state, play("7S"))
    state = apply(state, play("2S"))
    assert state.to_act == 2


def test_declining_a_chop_after_pass_moves_on():
    rules = Rules(n_players=3, chop_after_pass=True, first_lead_includes_lowest=False)
    state = table("4S 2S 9S", "3S 3C 3D 3H 9C", "5S 6S 7S", rules=rules)
    state = apply(state, play("4S"))
    state = apply(state, PASS)
    state = apply(state, play("7S"))
    state = apply(state, play("2S"))
    state = apply(state, PASS)
    assert state.to_act == 2
    state = apply(state, PASS)
    assert state.to_act == 0
    assert state.current is None


NO_FINISH = Rules(n_players=2, forbid_finish_on_two=True, first_lead_includes_lowest=False)


def test_finishing_on_a_two_is_allowed_by_default():
    assert play("2H") in legal_plays(apply(table("3S 9C", "2H"), play("3S")))


def test_forbidden_finish_on_a_two_forces_a_pass():
    state = apply(table("3S 9C", "2H", rules=NO_FINISH), play("3S"))
    assert legal_plays(state) == [PASS]


def test_forbidden_finish_blocks_multi_card_finishes_when_leading():
    assert legal_plays(table("2S 2H", "3S", rules=NO_FINISH)) == [play("2S"), play("2H")]


def test_forbidden_finish_lifts_when_a_two_is_the_only_lead():
    state = table("2H", "3S", rules=NO_FINISH)
    assert legal_plays(state) == [play("2H")]
    assert is_over(apply(state, play("2H")))


def test_forbidden_finish_ignores_non_twos():
    assert play("3S") in legal_plays(table("3S", "4S", rules=NO_FINISH))


FOUR_TWOS = "2S 2C 2D 2H 3S 4S 5S 6S 7C 8D 9H 10S JC"
SIX_PAIRS = "3S 3C 4S 4C 5S 5C 6S 6C 7S 7C 8S 8C 9S"
DRAGON = "3S 4C 5D 6H 7S 8C 9D 10H JS QC KD AH 2S"
PLAIN = "3S 3C 3D 4H 5S 7C 9D 10H JS QC KD AH 2S"


@pytest.mark.parametrize(
    ("hand", "expected"),
    [(FOUR_TWOS, "four_twos"), (SIX_PAIRS, "six_pairs"), (DRAGON, "dragon"), (PLAIN, None)],
)
def test_instant_win_hands(hand, expected):
    assert instant_win(parse_cards(hand)) == expected


def test_a_quad_counts_as_two_pairs():
    assert instant_win(parse_cards("3S 3C 3D 3H 4S 4C 5S 5C 6S 6C 7S 7C 9H")) == "six_pairs"


def _seed_with_instant_win(rules: Rules) -> int:
    return next(
        seed
        for seed in range(100_000)
        if any(instant_win(hand) for hand in new_game(rules, seed).hands)
    )


def test_instant_wins_end_the_game_on_the_deal():
    rules = Rules(instant_wins=True)
    seed = _seed_with_instant_win(rules)
    state = new_game(rules, seed)
    assert is_over(state)
    assert state.moves == ()
    assert instant_win(state.hands[placements(state)[0]])
    assert sorted(placements(state)) == [0, 1, 2, 3]


def test_instant_wins_are_ignored_by_default():
    seed = _seed_with_instant_win(Rules(instant_wins=True))
    assert not is_over(new_game(Rules(), seed))


def test_replay_rebuilds_the_game():
    state = new_game(Rules(), seed=4)
    moves = []
    while not is_over(state):
        chosen = legal_plays(state)[0]
        moves.append(Move(state.to_act, chosen))
        state = apply(state, chosen)
    assert replay(Rules(), 4, moves) == state


def test_replay_rejects_out_of_turn_moves():
    state = new_game(Rules(), seed=4)
    wrong_seat = (state.to_act + 1) % 4
    with pytest.raises(IllegalPlay):
        replay(Rules(), 4, [Move(wrong_seat, legal_plays(state)[0])])


def test_three_of_spades_opens_a_four_player_game():
    state = new_game(Rules(), seed=0)
    assert parse_card("3S") in state.hands[state.to_act]
