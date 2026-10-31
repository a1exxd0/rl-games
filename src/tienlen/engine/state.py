import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, Self

from tienlen.engine.cards import DECK, HAND_SIZE, Card, format_card, parse_card
from tienlen.engine.combos import Play
from tienlen.engine.rules import Rules


class Move(NamedTuple):
    seat: int
    play: Play

    def to_dict(self) -> dict[str, Any]:
        return {"seat": self.seat, "play": self.play.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(data["seat"], Play.from_dict(data["play"]))


def moves_to_list(moves: Iterable[Move]) -> list[dict[str, Any]]:
    return [move.to_dict() for move in moves]


def moves_from_list(data: Iterable[Mapping[str, Any]]) -> tuple[Move, ...]:
    return tuple(Move.from_dict(move) for move in data)


@dataclass(frozen=True, slots=True)
class GameState:
    rules: Rules
    seed: int | None
    hands: tuple[tuple[Card, ...], ...]
    undealt: tuple[Card, ...]
    first_leader: int
    to_act: int | None
    current: Play | None = None
    last_player: int | None = None
    passed: frozenset[int] = frozenset()
    placements: tuple[int, ...] = ()
    moves: tuple[Move, ...] = ()
    round_start: int = 0


@dataclass(frozen=True, slots=True)
class PlayerView:
    rules: Rules
    seat: int
    hand: tuple[Card, ...]
    hand_counts: tuple[int, ...]
    first_leader: int
    to_act: int | None
    current: Play | None
    last_player: int | None
    passed: frozenset[int]
    placements: tuple[int, ...]
    moves: tuple[Move, ...]
    round_start: int

    @property
    def trick(self) -> tuple[Move, ...]:
        return self.moves[self.round_start :]

    @property
    def played(self) -> tuple[Card, ...]:
        return tuple(sorted(card for move in self.moves for card in move.play.cards))

    def sample_world(self, rng: random.Random) -> GameState:
        hidden = sorted(set(DECK).difference(self.hand, self.played))
        lowest = self._sample_lowest_dealt(hidden, rng)
        pool = [card for card in hidden if card > lowest]
        rng.shuffle(pool)

        hands = []
        for seat, count in enumerate(self.hand_counts):
            if seat == self.seat:
                hands.append(self.hand)
                continue
            forced = [lowest] if seat == self.first_leader and lowest in hidden else []
            dealt = forced + [pool.pop() for _ in range(count - len(forced))]
            hands.append(tuple(sorted(dealt)))

        return GameState(
            rules=self.rules,
            seed=None,
            hands=tuple(hands),
            undealt=tuple(sorted(pool + [card for card in hidden if card < lowest])),
            first_leader=self.first_leader,
            to_act=self.to_act,
            current=self.current,
            last_player=self.last_player,
            passed=self.passed,
            placements=self.placements,
            moves=self.moves,
            round_start=self.round_start,
        )

    def _played_cards(self, *, by_leader: bool) -> list[Card]:
        leader = self.first_leader
        return [c for m in self.moves if (m.seat == leader) == by_leader for c in m.play.cards]

    def _lowest_is_known(self) -> bool:
        lead_was_shown = bool(self.moves) and self.rules.first_lead_includes_lowest
        leader_is_empty = not self.hand_counts[self.first_leader]
        return self.seat == self.first_leader or lead_was_shown or leader_is_empty

    def _sample_lowest_dealt(self, hidden: list[Card], rng: random.Random) -> Card:
        leader_cards = self._played_cards(by_leader=True)
        other_cards = self._played_cards(by_leader=False)
        (leader_cards if self.seat == self.first_leader else other_cards).extend(self.hand)

        leader_low = min(leader_cards, default=len(DECK))
        if self._lowest_is_known():
            return leader_low

        others_low = min(other_cards)
        n_undealt = len(DECK) - HAND_SIZE * self.rules.n_players
        options = [c for c in hidden[: n_undealt + 1] if c < min(others_low, leader_low)]
        if leader_low < others_low and sum(c < leader_low for c in hidden) <= n_undealt:
            options.append(leader_low)
        return rng.choice(options)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": self.rules.to_dict(),
            "seat": self.seat,
            "hand": [format_card(card) for card in self.hand],
            "hand_counts": list(self.hand_counts),
            "first_leader": self.first_leader,
            "to_act": self.to_act,
            "current": self.current.to_dict() if self.current else None,
            "last_player": self.last_player,
            "passed": sorted(self.passed),
            "placements": list(self.placements),
            "moves": moves_to_list(self.moves),
            "round_start": self.round_start,
            "played": [format_card(card) for card in self.played],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        current = data["current"]
        return cls(
            rules=Rules.from_dict(data["rules"]),
            seat=data["seat"],
            hand=tuple(sorted(parse_card(card) for card in data["hand"])),
            hand_counts=tuple(data["hand_counts"]),
            first_leader=data["first_leader"],
            to_act=data["to_act"],
            current=Play.from_dict(current) if current else None,
            last_player=data["last_player"],
            passed=frozenset(data["passed"]),
            placements=tuple(data["placements"]),
            moves=moves_from_list(data["moves"]),
            round_start=data["round_start"],
        )
