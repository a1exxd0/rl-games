# Engine API

Import everything from `tienlen.engine`. It uses only the standard library and imports under Pyodide, so the web app, RL env, and CLI all run the same rules. Do not import `tienlen.cli`, `tienlen.registry`, or anything with third-party dependencies from browser code. Install the wheel with `deps=False` there.

## Usage

```python
import random
from tienlen.engine import Rules, apply, is_over, legal_plays, new_game, placements, replay, view

rules = Rules(n_players=2)
state = new_game(rules, seed=7)
rng = random.Random(0)

while not is_over(state):
    seat = state.to_act
    seen = view(state, seat)  # what a bot may see
    state = apply(state, rng.choice(legal_plays(state)))

print(placements(state))  # seats in finishing order
assert replay(rules, 7, state.moves) == state
```

## Cards

A `Card` is an `int` in 0..51: `rank * 4 + suit`, so `<` compares strength. Ranks run `3 4 5 6 7 8 9 10 J Q K A 2`, suits run `S C D H`. Text form is `"3S"`, `"10D"`, `"2H"`; `T` is accepted for parsing.

```python
DECK: tuple[Card, ...]; RANKS; SUITS; TWO_RANK; HAND_SIZE
make_card(rank: int, suit: int) -> Card
rank_of(card) -> int; suit_of(card) -> int
parse_card(text: str) -> Card                  # ValueError on bad input
format_card(card) -> str
parse_cards("3S 4H") -> tuple[Card, ...]       # sorted; spaces or commas
format_cards(cards) -> str
```

## Plays

```python
class ComboType(StrEnum):  # PASS SINGLE PAIR TRIPLE QUAD SEQUENCE DOUBLE_SEQUENCE

@dataclass(frozen=True)
class Play:
    cards: tuple[Card, ...]   # sorted ascending
    type: ComboType
    Play.of(cards) -> Play    # sorts and classifies; raises InvalidPlay
    is_pass: bool
    top: Card                 # highest card; undefined on PASS
    to_dict() / Play.from_dict(d)   # {"type": "pair", "cards": ["3S", "3H"]}

PASS: Play
classify(cards) -> ComboType | None          # () is PASS; None if not a combo
beats(play, current) -> bool                 # current must be non-pass
is_chop(play, current) -> bool               # true only for the chop table
enumerate_plays(hand) -> list[Play]          # all combos, ignoring game state, no PASS
```

Same type and same card count compare on `top`. The chop table is exact, with nothing implied beyond it:

| Play | Beats |
|---|---|
| Quad | Single 2 |
| 3-pair double sequence | Single 2 |
| 4-pair double sequence | Pair of 2s, any quad |
| 5-pair double sequence | Triple 2s |

## Rules

```python
@dataclass(frozen=True)
class Rules:
    n_players: int = 4  # 2-4; always 13 cards each
    lock_on_pass: bool = True  # a pass locks you out until the round ends
    chop_after_pass: bool = False  # a locked seat holding a chop gets a turn
    instant_wins: bool = False  # four 2s, six pairs, or dragon wins on the deal
    first_lead_includes_lowest: bool = True  # first lead must contain the lowest dealt card
    forbid_finish_on_two: bool = False  # a play that empties your hand cannot be 2s
    rules_id: str  # stable 12-char hash of the config
    to_dict() / Rules.from_dict(d)
```

Store `rules_id` with anything that gets compared, since ratings mean nothing across rule sets. Also store `ENGINE_VERSION`, a string bumped when engine behaviour changes.

## Game

All functions are pure and states are immutable, so search can branch without copying.

```python
new_game(rules: Rules, seed: int) -> GameState
legal_plays(state) -> list[Play]             # PASS is last when legal; [] once over
is_legal(state, play) -> bool
apply(state, play) -> GameState              # raises IllegalPlay
view(state, seat) -> PlayerView
is_over(state) -> bool
placements(state) -> tuple[int, ...]         # finishing order; partial until over
replay(rules, seed, moves) -> GameState      # raises IllegalPlay if moves do not fit
instant_win(hand) -> "four_twos" | "six_pairs" | "dragon" | None
```

`GameState` holds `rules, seed, hands, undealt, first_leader, to_act, current, last_player, passed, placements, moves, round_start`. `to_act` is `None` once the game is over. It contains hidden cards and the deal seed, so **never pass a `GameState` or the deal seed to a bot**.

`moves` is a tuple of `Move(seat: int, play: Play)`, passes included. Save a game as `rules.to_dict()`, `seed`, and `moves_to_list(moves)`; load with `moves_from_list(data)` and `replay`.

## PlayerView

What one seat may know. It has no opponent cards, undealt cards, or seed.

```python
@dataclass(frozen=True)
class PlayerView:
    rules: Rules
    seat: int
    hand: tuple[Card, ...]
    hand_counts: tuple[int, ...]    # per seat
    first_leader: int
    to_act: int | None
    current: Play | None            # None means the seat to act is leading
    last_player: int | None         # who made `current`
    passed: frozenset[int]
    placements: tuple[int, ...]
    moves: tuple[Move, ...]         # full public history
    round_start: int
    trick: tuple[Move, ...]         # moves of the current round
    played: tuple[Card, ...]        # sorted, all cards played so far
    sample_world(rng: random.Random) -> GameState
    to_dict() / PlayerView.from_dict(d)
```

`sample_world` deals hidden cards to match the hand counts and the first-lead rule, for tree search. The result is a normal `GameState` with `seed=None`, so `legal_plays` and `apply` work on it. It does not infer anything from past passes, and it does not exclude instant-win hands.

`to_dict` is JSON-safe, with cards as strings. It is the payload for the web worker bridge.

## Bots

A bot is a Python file defining `create_bot(artifact_dir: Path | None)`, which returns a `TienLenBot` from `tienlen.bots`. `artifact_dir` holds the version's stored files, such as weights.

```python
class TienLenBot(Protocol):
    def reset(self, seat: int, rules: Rules, seed: int) -> None: ...
    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play: ...
```

Runners must give `reset` a bot-specific seed, never the deal seed. `legal` is never empty. See `bots/random/bot.py` and `bots/greedy/bot.py`.

## Rule decisions

- Each game is independent: `new_game` always gives the lead to the holder of the lowest dealt card, including after a previous game. A series where the winner leads next is not modelled.
- Turn order walks the ring from the last mover. Each seat gets one chance per play, and reaching `last_player` ends the round, even if that seat is out of cards. The lead then goes to `last_player` or the next seat holding cards.
- `lock_on_pass=False`: `passed` is cleared on every play, so a seat that passed may play on its next turn.
- `chop_after_pass=True`: a locked seat holding a chop for `current` gets a turn with only its chops and `PASS`. Playing one unlocks it. A higher same-type bomb is not a chop, so it does not reopen a locked seat.
- `forbid_finish_on_two=True`: a play that empties your hand and contains a 2 is illegal, except when leading with a single 2 as your last card, since there is no other move.
- `instant_wins=True`: checked at the deal, starting from the first leader in turn order. The first match wins, the rest place in ring order from the winner, and `moves` stays empty. A quad counts as two pairs.
