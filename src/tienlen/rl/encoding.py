from collections.abc import Iterable, Sequence

import numpy as np

from tienlen.engine import (
    DECK,
    HAND_SIZE,
    RANKS,
    SUITS,
    TWO_RANK,
    Card,
    ComboType,
    Play,
    PlayerView,
    is_chop,
    rank_of,
    suit_of,
)

N_CARDS = len(DECK)
N_RANKS = len(RANKS)
MAX_SEATS = 4
PLAYER_COUNTS = (2, 3, 4)
COMBO_TYPES = tuple(ComboType)
RULE_FLAGS = (
    "lock_on_pass",
    "chop_after_pass",
    "instant_wins",
    "first_lead_includes_lowest",
    "forbid_finish_on_two",
)

STATE_DIM = 3 * N_CARDS + 2 * N_RANKS + 3 * MAX_SEATS + 1 + len(PLAYER_COUNTS) + len(RULE_FLAGS)
ACTION_DIM = N_CARDS + len(COMBO_TYPES) + 9


def _card_bits(cards: Iterable[Card]) -> np.ndarray:
    bits = np.zeros(N_CARDS, dtype=np.float32)
    bits[list(cards)] = 1
    return bits


def _rank_shares(bits: np.ndarray) -> np.ndarray:
    return bits.reshape(N_RANKS, len(SUITS)).sum(axis=1) / len(SUITS)


def _seat_features(view: PlayerView) -> np.ndarray:
    n_players = view.rules.n_players
    seats = [(view.seat + offset) % n_players for offset in range(n_players)]
    features = np.zeros((3, MAX_SEATS), dtype=np.float32)
    features[0, :n_players] = [view.hand_counts[seat] / HAND_SIZE for seat in seats]
    features[1, :n_players] = [seat in view.passed for seat in seats]
    features[2, :n_players] = [seat == view.last_player for seat in seats]
    return features.ravel()


def _table_features(view: PlayerView) -> np.ndarray:
    players = [view.rules.n_players == count for count in PLAYER_COUNTS]
    rules = [getattr(view.rules, flag) for flag in RULE_FLAGS]
    return np.array([view.current is None, *players, *rules], dtype=np.float32)


def encode_state(view: PlayerView) -> np.ndarray:
    hand = _card_bits(view.hand)
    played = _card_bits(view.played)
    current = _card_bits(view.current.cards if view.current else ())
    return np.concatenate(
        [
            hand,
            played,
            current,
            _rank_shares(hand),
            _rank_shares(played),
            _seat_features(view),
            _table_features(view),
        ]
    )


def _lone_ranks(rank_counts: Sequence[int], cards: Iterable[Card]) -> int:
    remaining = list(rank_counts)
    for card in cards:
        remaining[rank_of(card)] -= 1
    return sum(count == 1 for count in remaining)


def _play_summary(view: PlayerView, play: Play, rank_counts: list[int]) -> list[float]:
    left = len(view.hand) - len(play.cards)
    chop = bool(view.current and not play.is_pass and is_chop(play, view.current))
    twos = sum(rank_of(card) == TWO_RANK for card in play.cards)
    return [
        len(play.cards) / HAND_SIZE,
        chop,
        twos / len(SUITS),
        left / HAND_SIZE,
        left == 0,
        _lone_ranks(rank_counts, play.cards) / N_RANKS,
    ]


def _top_features(play: Play, unseen_above: np.ndarray) -> tuple[float, float, float]:
    if play.is_pass:
        return 0.0, 0.0, 0.0
    return (
        rank_of(play.top) / (N_RANKS - 1),
        suit_of(play.top) / (len(SUITS) - 1),
        float(unseen_above[play.top]),
    )


def _unseen_above(view: PlayerView) -> np.ndarray:
    unseen = 1 - _card_bits((*view.hand, *view.played))
    return (unseen[::-1].cumsum()[::-1] - unseen) / N_CARDS


def encode_actions(view: PlayerView, legal: Sequence[Play]) -> np.ndarray:
    rank_counts = [0] * N_RANKS
    for card in view.hand:
        rank_counts[rank_of(card)] += 1
    unseen_above = _unseen_above(view)

    encoded = np.zeros((len(legal), ACTION_DIM), dtype=np.float32)
    encoded[:, N_CARDS + len(COMBO_TYPES) :] = [
        [*_play_summary(view, play, rank_counts), *_top_features(play, unseen_above)]
        for play in legal
    ]
    for row, play in enumerate(legal):
        encoded[row, list(play.cards)] = 1
        encoded[row, N_CARDS + COMBO_TYPES.index(play.type)] = 1
    return encoded
