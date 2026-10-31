from itertools import combinations

import pytest
from helpers import play
from hypothesis import given, settings
from hypothesis import strategies as st

from tienlen.engine import (
    DECK,
    PASS,
    ComboType,
    InvalidPlay,
    Play,
    beats,
    classify,
    enumerate_plays,
    is_chop,
    parse_cards,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ComboType.PASS),
        ("3S", ComboType.SINGLE),
        ("2H", ComboType.SINGLE),
        ("5S 5H", ComboType.PAIR),
        ("2S 2H", ComboType.PAIR),
        ("9S 9C 9H", ComboType.TRIPLE),
        ("JS JC JD JH", ComboType.QUAD),
        ("3S 4C 5D", ComboType.SEQUENCE),
        ("3S 4S 5S 6S 7S 8S 9S 10S JS QS KS AS", ComboType.SEQUENCE),
        ("QH KH AH", ComboType.SEQUENCE),
        ("3S 3C 4S 4C 5S 5C", ComboType.DOUBLE_SEQUENCE),
        ("QS QC KS KC AS AC", ComboType.DOUBLE_SEQUENCE),
        ("3S 3C 4S 4C 5S 5C 6S 6C", ComboType.DOUBLE_SEQUENCE),
    ],
)
def test_classify_recognises_combos(text, expected):
    assert classify(parse_cards(text)) is expected


@pytest.mark.parametrize(
    "text",
    [
        "3S 4C",
        "3S 5C 6D",
        "KS AS 2S",
        "AS 2S 3S",
        "3S 3C 4S 4C",
        "KS KC AS AC 2S 2C",
        "3S 3C 3D 4S 4C 4D",
        "3S 3C 5S 5C 6S 6C",
        "3S 3C 4S",
    ],
)
def test_classify_rejects_non_combos(text):
    assert classify(parse_cards(text)) is None


def test_classify_rejects_duplicates():
    assert classify((0, 0)) is None


def test_play_of_sorts_and_rejects():
    assert play("4C 3S 5D").cards == parse_cards("3S 4C 5D")
    with pytest.raises(InvalidPlay):
        Play.of(parse_cards("3S 5C"))


@pytest.mark.parametrize(
    ("high", "low"),
    [
        ("4S", "3H"),
        ("3H", "3S"),
        ("2S", "AH"),
        ("5S 5H", "5C 5D"),
        ("6S 6C", "5D 5H"),
        ("4S 4C 4D", "3D 3C 3H"),
        ("4S 4C 4D 4H", "3S 3C 3D 3H"),
        ("4S 5S 6S", "3H 4H 5H"),
        ("3S 4S 5H", "3C 4D 5D"),
        ("4S 4C 5S 5C 6S 6C", "3D 3H 4D 4H 5D 5H"),
    ],
)
def test_same_type_compares_on_top_card(high, low):
    assert beats(play(high), play(low))
    assert not beats(play(low), play(high))


def test_equal_length_is_required_for_sequences():
    assert not beats(play("7S 8S 9S 10S"), play("3S 4S 5S"))
    assert not beats(play("7S 8S 9S"), play("3S 4S 5S 6S"))


def test_different_types_do_not_beat_each_other():
    assert not beats(play("4S 4C"), play("3S"))
    assert not beats(play("4S 5C 6D"), play("3S 3C 3D"))


def test_pass_beats_nothing():
    assert not beats(PASS, play("3S"))


THREE_PAIRS = "3S 3C 4S 4C 5S 5C"
FOUR_PAIRS = "3S 3C 4S 4C 5S 5C 6S 6C"
FIVE_PAIRS = "3S 3C 4S 4C 5S 5C 6S 6C 7S 7C"
QUAD = "3S 3C 3D 3H"


@pytest.mark.parametrize(
    ("chop", "target"),
    [
        (QUAD, "2S"),
        (QUAD, "2H"),
        (THREE_PAIRS, "2H"),
        (FOUR_PAIRS, "2S 2H"),
        (FOUR_PAIRS, "9S 9C 9D 9H"),
        (FOUR_PAIRS, "2S 2C 2D 2H"),
        (FIVE_PAIRS, "2S 2D 2H"),
    ],
)
def test_chops(chop, target):
    assert is_chop(play(chop), play(target))
    assert beats(play(chop), play(target))


@pytest.mark.parametrize(
    ("not_chop", "target"),
    [
        (QUAD, "AH"),
        (QUAD, "2S 2H"),
        (THREE_PAIRS, "AH"),
        (THREE_PAIRS, "2S 2H"),
        (THREE_PAIRS, "9S 9C 9D 9H"),
        (FOUR_PAIRS, "2H"),
        (FOUR_PAIRS, "AS AH"),
        (FOUR_PAIRS, "2S 2D 2H"),
        (FIVE_PAIRS, "AS AC AH"),
        (FIVE_PAIRS, "2H"),
        (FIVE_PAIRS, "2S 2H"),
        (FIVE_PAIRS, "9S 9C 9D 9H"),
    ],
)
def test_outside_the_chop_table_nothing_chops(not_chop, target):
    assert not beats(play(not_chop), play(target))


def test_same_type_bomb_is_not_a_chop():
    assert beats(play("9S 9C 9D 9H"), play(QUAD))
    assert not is_chop(play("9S 9C 9D 9H"), play(QUAD))


def test_a_higher_chop_answers_a_chop():
    assert beats(play("4S 4C 5S 5C 6S 6C"), play("3D 3H 4D 4H 5D 5H"))


hands = st.lists(st.sampled_from(DECK), min_size=0, max_size=9, unique=True)


@settings(max_examples=150)
@given(hands)
def test_enumeration_matches_brute_force(hand):
    expected = {
        subset
        for size in range(1, len(hand) + 1)
        for subset in combinations(sorted(hand), size)
        if classify(subset) is not None
    }
    plays = enumerate_plays(hand)
    assert {p.cards for p in plays} == expected
    assert len(plays) == len(expected)
    assert all(p.type is classify(p.cards) for p in plays)


def test_enumeration_on_a_full_hand():
    hand = parse_cards("3S 3C 3D 3H 4S 4C 5S 5C 5D 6S 6C 7H 2S")
    plays = enumerate_plays(hand)
    assert play("3S 3C 4S 4C 5S 5C 6S 6C") in plays
    assert play("3S 3C 3D 3H") in plays
    assert all(p != PASS for p in plays)


def test_play_dict_round_trip():
    original = play("3S 4C 5D")
    assert original.to_dict() == {"type": "sequence", "cards": ["3S", "4C", "5D"]}
    assert Play.from_dict(original.to_dict()) == original
    assert Play.from_dict(PASS.to_dict()) == PASS


def test_play_from_dict_rejects_a_wrong_type():
    with pytest.raises(InvalidPlay):
        Play.from_dict({"type": "pair", "cards": ["3S"]})
