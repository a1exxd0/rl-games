import importlib.util
import json
import secrets
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

from tienlen.bots import TienLenBot
from tienlen.engine import (
    GameState,
    Play,
    Rules,
    apply,
    format_card,
    is_over,
    legal_plays,
    moves_from_list,
    moves_to_list,
    new_game,
    parse_cards,
    replay,
    view,
)

HUMAN = "human"


def load_bot(bots_dir: Path, name: str) -> TienLenBot:
    folder = bots_dir / name
    spec = importlib.util.spec_from_file_location(f"bot_{name}", folder / "bot.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bot: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    has_artifacts = any(path.is_file() and path.name != "bot.py" for path in folder.iterdir())
    return module.create_bot(folder if has_artifacts else None)


def snapshot(
    state: GameState,
    seats: Sequence[str],
    perspective: int,
    open_hands: bool,
    legal: Sequence[Play] = (),
) -> dict[str, Any]:
    over = is_over(state)
    return {
        "view": view(state, perspective).to_dict(),
        "hands": [
            [format_card(card) for card in hand] if open_hands or seat == perspective else None
            for seat, hand in enumerate(state.hands)
        ],
        "legal": [play.to_dict() for play in legal],
        "seats": list(seats),
        "human": None if open_hands else perspective,
        "over": over,
        "rules_id": state.rules.rules_id,
        "seed": state.seed if open_hands or over else None,
    }


class Game:
    def __init__(self, rules: Rules, seats: Sequence[str], seed: int, bots_dir: Path) -> None:
        if len(seats) != rules.n_players:
            raise ValueError(f"expected {rules.n_players} seats, got {len(seats)}")
        self.seats = list(seats)
        self.state = new_game(rules, seed)
        self.human = seats.index(HUMAN) if HUMAN in seats else None
        self.bots = {
            seat: load_bot(bots_dir, name) for seat, name in enumerate(seats) if name != HUMAN
        }
        for seat, bot in self.bots.items():
            bot.reset(seat, rules, secrets.randbits(32))

    def step(self) -> None:
        seat = self.state.to_act
        if seat is None or seat == self.human:
            return
        play = self.bots[seat].act(view(self.state, seat), legal_plays(self.state))
        self.state = apply(self.state, play)

    def play(self, cards: Sequence[str]) -> None:
        if self.human is None or self.state.to_act != self.human:
            raise ValueError("it is not your turn")
        self.state = apply(self.state, Play.of(parse_cards(" ".join(cards))))

    def snapshot(self) -> dict[str, Any]:
        if self.human is None:
            return snapshot(self.state, self.seats, 0, True)
        my_turn = self.state.to_act == self.human
        legal = legal_plays(self.state) if my_turn else ()
        return snapshot(self.state, self.seats, self.human, False, legal)

    def export(self) -> dict[str, Any]:
        if not is_over(self.state):
            raise ValueError("the game is not over")
        return {
            "rules": self.state.rules.to_dict(),
            "seed": self.state.seed,
            "moves": moves_to_list(self.state.moves),
            "seats": self.seats,
        }


class Replay:
    def __init__(self, data: dict[str, Any]) -> None:
        self.rules = Rules.from_dict(data["rules"])
        self.seed = data["seed"]
        self.moves = moves_from_list(data["moves"])
        names = data.get("seats") or [f"Seat {seat}" for seat in range(self.rules.n_players)]
        self.seats = [str(name) for name in names]
        self.final = replay(self.rules, self.seed, self.moves)
        self.cursor = 0

    def seek(self, cursor: int) -> None:
        self.cursor = max(0, min(cursor, len(self.moves)))

    def snapshot(self) -> dict[str, Any]:
        state = replay(self.rules, self.seed, self.moves[: self.cursor])
        return snapshot(state, self.seats, 0, True) | {
            "cursor": self.cursor,
            "total": len(self.moves),
        }


class App:
    def __init__(self, bots_dir: Path = Path("/bots")) -> None:
        self.bots_dir = bots_dir
        self.table: Game | Replay | None = None
        self.commands = {
            "schema": self.schema,
            "new_game": self.new_game,
            "step": self.step,
            "play": self.play,
            "load_replay": self.load_replay,
            "seek": self.seek,
            "export": self.export,
        }

    def schema(self) -> dict[str, bool]:
        return {f.name: f.default for f in fields(Rules) if isinstance(f.default, bool)}

    def new_game(
        self, rules: dict[str, Any], seats: Sequence[str], seed: int | None = None
    ) -> dict[str, Any]:
        deal = secrets.randbits(32) if seed is None else seed
        self.table = Game(Rules.from_dict(rules), seats, deal, self.bots_dir)
        return self.table.snapshot()

    def _game(self) -> Game:
        if isinstance(self.table, Game):
            return self.table
        raise ValueError("no game is open")

    def _replay(self) -> Replay:
        if isinstance(self.table, Replay):
            return self.table
        raise ValueError("no replay is open")

    def step(self) -> dict[str, Any]:
        game = self._game()
        game.step()
        return game.snapshot()

    def play(self, cards: Sequence[str]) -> dict[str, Any]:
        game = self._game()
        game.play(cards)
        return game.snapshot()

    def load_replay(self, replay: dict[str, Any]) -> dict[str, Any]:
        self.table = Replay(replay)
        return self.table.snapshot()

    def seek(self, cursor: int) -> dict[str, Any]:
        replay = self._replay()
        replay.seek(cursor)
        return replay.snapshot()

    def export(self) -> dict[str, Any]:
        return self._game().export()

    def handle(self, request: str) -> str:
        message = json.loads(request)
        try:
            result = self.commands[message["type"]](**message.get("payload", {}))
        except Exception as error:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"})
        return json.dumps({"ok": True, "result": result})
