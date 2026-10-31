from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations, pairwise, product
from typing import Any, Self

from tienlen.engine.cards import TWO_RANK, Card, format_card, parse_card, rank_of

MIN_SEQUENCE = 3
MIN_DOUBLE_SEQUENCE_PAIRS = 3


class ComboType(StrEnum):
    PASS = "pass"
    SINGLE = "single"
    PAIR = "pair"
    TRIPLE = "triple"
    QUAD = "quad"
    SEQUENCE = "sequence"
    DOUBLE_SEQUENCE = "double_sequence"


SAME_RANK_TYPES = {
    1: ComboType.SINGLE,
    2: ComboType.PAIR,
    3: ComboType.TRIPLE,
    4: ComboType.QUAD,
}


class InvalidPlay(ValueError):
    pass


def _is_run(ranks: list[int]) -> bool:
    return ranks[-1] < TWO_RANK and all(b - a == 1 for a, b in pairwise(ranks))


def classify(cards: Iterable[Card]) -> ComboType | None:
    ordered = sorted(cards)
    if len(set(ordered)) != len(ordered):
        return None
    if not ordered:
        return ComboType.PASS

    ranks = [rank_of(card) for card in ordered]
    if len(set(ranks)) == 1:
        return SAME_RANK_TYPES.get(len(ranks))
    if len(ranks) >= MIN_SEQUENCE and _is_run(ranks):
        return ComboType.SEQUENCE

    pair_count = len(ranks) // 2
    is_pairs = len(ranks) % 2 == 0 and ranks[::2] == ranks[1::2]
    if is_pairs and pair_count >= MIN_DOUBLE_SEQUENCE_PAIRS and _is_run(ranks[::2]):
        return ComboType.DOUBLE_SEQUENCE
    return None


@dataclass(frozen=True, slots=True)
class Play:
    cards: tuple[Card, ...]
    type: ComboType

    @classmethod
    def of(cls, cards: Iterable[Card]) -> Self:
        ordered = tuple(sorted(cards))
        combo = classify(ordered)
        if combo is None:
            raise InvalidPlay(f"not a combo: {[format_card(card) for card in ordered]}")
        return cls(ordered, combo)

    @property
    def is_pass(self) -> bool:
        return self.type is ComboType.PASS

    @property
    def top(self) -> Card:
        return self.cards[-1]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "cards": [format_card(card) for card in self.cards]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        play = cls.of(parse_card(card) for card in data["cards"])
        if play.type != data["type"]:
            raise InvalidPlay(f"{data['cards']} is a {play.type}, not a {data['type']}")
        return play


PASS = Play((), ComboType.PASS)


def is_chop(play: Play, current: Play) -> bool:
    top_is_two = rank_of(current.top) == TWO_RANK
    match play.type, len(play.cards):
        case ComboType.QUAD, _:
            return current.type is ComboType.SINGLE and top_is_two
        case ComboType.DOUBLE_SEQUENCE, 6:
            return current.type is ComboType.SINGLE and top_is_two
        case ComboType.DOUBLE_SEQUENCE, 8:
            return current.type is ComboType.QUAD or (current.type is ComboType.PAIR and top_is_two)
        case ComboType.DOUBLE_SEQUENCE, 10:
            return current.type is ComboType.TRIPLE and top_is_two
    return False


def beats(play: Play, current: Play) -> bool:
    if play.is_pass:
        return False
    if play.type is current.type and len(play.cards) == len(current.cards):
        return play.top > current.top
    return is_chop(play, current)


def _consecutive_groups(
    groups: dict[int, list[tuple[Card, ...]]], min_length: int
) -> Iterator[list[list[tuple[Card, ...]]]]:
    for start in groups:
        run = []
        for rank in range(start, TWO_RANK):
            if rank not in groups:
                break
            run.append(groups[rank])
            if len(run) >= min_length:
                yield list(run)


def _joined(run: list[list[tuple[Card, ...]]], combo: ComboType) -> list[Play]:
    return [Play(sum(chosen, ()), combo) for chosen in product(*run)]


def _run_plays(
    groups: dict[int, list[tuple[Card, ...]]], min_length: int, combo: ComboType
) -> list[Play]:
    runs = _consecutive_groups(groups, min_length)
    return [play for run in runs for play in _joined(run, combo)]


def _same_rank_plays(cards: list[Card]) -> list[Play]:
    return [
        Play(chosen, SAME_RANK_TYPES[size])
        for size in range(1, len(cards) + 1)
        for chosen in combinations(cards, size)
    ]


def enumerate_plays(hand: Iterable[Card]) -> list[Play]:
    by_rank: dict[int, list[Card]] = {}
    for card in sorted(hand):
        by_rank.setdefault(rank_of(card), []).append(card)

    singles = {rank: [(card,) for card in cards] for rank, cards in by_rank.items()}
    pairs = {
        rank: list(combinations(cards, 2)) for rank, cards in by_rank.items() if len(cards) >= 2
    }
    return [
        *(play for cards in by_rank.values() for play in _same_rank_plays(cards)),
        *_run_plays(singles, MIN_SEQUENCE, ComboType.SEQUENCE),
        *_run_plays(pairs, MIN_DOUBLE_SEQUENCE_PAIRS, ComboType.DOUBLE_SEQUENCE),
    ]
