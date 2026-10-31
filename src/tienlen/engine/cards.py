from collections.abc import Iterable

Card = int

RANKS = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")
SUITS = ("S", "C", "D", "H")
TWO_RANK = RANKS.index("2")
DECK: tuple[Card, ...] = tuple(range(len(RANKS) * len(SUITS)))
HAND_SIZE = 13


def make_card(rank: int, suit: int) -> Card:
    return rank * len(SUITS) + suit


def rank_of(card: Card) -> int:
    return card // len(SUITS)


def suit_of(card: Card) -> int:
    return card % len(SUITS)


def parse_card(text: str) -> Card:
    rank, suit = text[:-1].upper(), text[-1:].upper()
    rank = "10" if rank == "T" else rank
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"invalid card: {text!r}")
    return make_card(RANKS.index(rank), SUITS.index(suit))


def format_card(card: Card) -> str:
    return RANKS[rank_of(card)] + SUITS[suit_of(card)]


def parse_cards(text: str) -> tuple[Card, ...]:
    return tuple(sorted(parse_card(token) for token in text.replace(",", " ").split()))


def format_cards(cards: Iterable[Card]) -> str:
    return " ".join(format_card(card) for card in cards)
