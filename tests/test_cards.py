import pytest

from tienlen.engine import DECK, format_card, format_cards, parse_card, parse_cards, rank_of


def test_rank_order_is_three_low_two_high():
    assert parse_card("3S") == 0
    assert parse_card("2H") == 51
    assert parse_card("AH") < parse_card("2S")
    assert parse_card("10S") > parse_card("9H")


def test_suit_order_within_a_rank():
    assert parse_card("7S") < parse_card("7C") < parse_card("7D") < parse_card("7H")


def test_every_card_round_trips():
    assert [parse_card(format_card(card)) for card in DECK] == list(DECK)


def test_ten_accepts_t():
    assert parse_card("TD") == parse_card("10D")


def test_lowercase_is_accepted():
    assert parse_card("qh") == parse_card("QH")


@pytest.mark.parametrize("text", ["", "1S", "3X", "S3", "11H"])
def test_invalid_cards_raise(text):
    with pytest.raises(ValueError):
        parse_card(text)


def test_parse_cards_sorts_and_splits_on_commas():
    assert format_cards(parse_cards("2H, 3S 10D")) == "3S 10D 2H"


def test_ranks_are_thirteen_groups_of_four():
    assert [rank_of(card) for card in DECK[:5]] == [0, 0, 0, 0, 1]
