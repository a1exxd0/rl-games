import importlib.util
import json
from pathlib import Path

import pytest

from tienlen.engine import Rules, new_game, parse_card

ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"


def load_session():
    spec = importlib.util.spec_from_file_location("session", ROOT / "web" / "py" / "session.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load_session()


def call(app, type_, **payload):
    reply = json.loads(app.handle(json.dumps({"type": type_, "payload": payload})))
    assert reply["ok"], reply
    return reply["result"]


def rules(n_players=2, **toggles):
    return Rules(n_players=n_players, **toggles).to_dict()


def finish(app, snapshot):
    while not snapshot["over"]:
        snapshot = call(app, "step")
    return snapshot


def human_card_turns(app, snapshot):
    while not snapshot["over"]:
        if snapshot["legal"]:
            snapshot = call(app, "play", cards=snapshot["legal"][0]["cards"])
        else:
            snapshot = call(app, "step")
    return snapshot


@pytest.fixture
def app():
    return session.App(BOTS)


def test_schema_lists_boolean_toggles(app):
    schema = call(app, "schema")
    assert schema["lock_on_pass"] is True
    assert schema["instant_wins"] is False
    assert "n_players" not in schema


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_bot_game_runs_to_completion_with_open_hands(app, n_players):
    seats = (["greedy", "random"] * 2)[:n_players]
    snapshot = call(app, "new_game", rules=rules(n_players), seats=seats, seed=5)
    assert all(hand is not None for hand in snapshot["hands"])
    assert snapshot["human"] is None
    assert snapshot["legal"] == []

    final = finish(app, snapshot)
    assert len(final["view"]["placements"]) == n_players
    assert final["view"]["to_act"] is None


def test_human_game_hides_opponent_hands_and_seed(app):
    snapshot = call(app, "new_game", rules=rules(), seats=["human", "greedy"], seed=5)
    assert snapshot["hands"][1] is None
    assert len(snapshot["hands"][0]) == 13
    assert snapshot["seed"] is None
    assert snapshot["human"] == 0


def test_human_game_reaches_the_end(app):
    snapshot = call(app, "new_game", rules=rules(), seats=["human", "random"], seed=3)
    final = human_card_turns(app, snapshot)
    assert final["seed"] == 3
    assert any(move["seat"] == 0 for move in final["view"]["moves"])


def test_step_does_not_move_for_the_human(app):
    snapshot = call(app, "new_game", rules=rules(), seats=["human", "greedy"], seed=5)
    while not snapshot["legal"]:
        snapshot = call(app, "step")
    assert call(app, "step")["view"]["moves"] == snapshot["view"]["moves"]


def test_human_can_pass_with_empty_cards(app):
    snapshot = call(app, "new_game", rules=rules(), seats=["human", "greedy"], seed=1)
    while not any(play["type"] == "pass" for play in snapshot["legal"]):
        snapshot = (
            call(app, "play", cards=snapshot["legal"][0]["cards"])
            if snapshot["legal"]
            else call(app, "step")
        )
    moved = call(app, "play", cards=[])
    assert moved["view"]["moves"][-1]["play"]["type"] == "pass"


def test_errors_are_reported_not_raised(app):
    call(app, "new_game", rules=rules(), seats=["human", "greedy"], seed=5)
    reply = json.loads(app.handle(json.dumps({"type": "play", "payload": {"cards": ["2H", "3S"]}})))
    assert reply["ok"] is False
    assert reply["error"]


def test_play_out_of_turn_is_rejected(app):
    snapshot = call(app, "new_game", rules=rules(), seats=["human", "greedy"], seed=5)
    while snapshot["legal"]:
        snapshot = call(app, "play", cards=snapshot["legal"][0]["cards"])
    reply = json.loads(app.handle(json.dumps({"type": "play", "payload": {"cards": []}})))
    assert reply["ok"] is False


def test_wrong_seat_count_is_rejected(app):
    reply = json.loads(
        app.handle(
            json.dumps({"type": "new_game", "payload": {"rules": rules(4), "seats": ["greedy"]}})
        )
    )
    assert reply["ok"] is False


def test_export_requires_a_finished_game(app):
    call(app, "new_game", rules=rules(), seats=["greedy", "random"], seed=5)
    reply = json.loads(app.handle(json.dumps({"type": "export"})))
    assert reply["ok"] is False


def test_replay_round_trip_steps_through_every_move(app):
    live = finish(app, call(app, "new_game", rules=rules(3), seats=["greedy"] * 3, seed=9))
    exported = call(app, "export")

    start = call(app, "load_replay", replay=exported)
    assert start["cursor"] == 0
    assert start["total"] == len(live["view"]["moves"])
    assert start["view"]["moves"] == []

    end = call(app, "seek", cursor=start["total"])
    assert end["view"]["moves"] == live["view"]["moves"]
    assert end["view"]["placements"] == live["view"]["placements"]
    assert end["seats"] == ["greedy"] * 3


def test_replay_seek_clamps_and_labels_default_seats(app):
    finish(app, call(app, "new_game", rules=rules(), seats=["greedy", "random"], seed=2))
    exported = call(app, "export")
    del exported["seats"]

    snapshot = call(app, "load_replay", replay=exported)
    assert snapshot["seats"] == ["Seat 0", "Seat 1"]
    assert call(app, "seek", cursor=-5)["cursor"] == 0
    assert call(app, "seek", cursor=10**6)["cursor"] == snapshot["total"]


def test_replay_rejects_moves_that_do_not_fit(app):
    finish(app, call(app, "new_game", rules=rules(), seats=["greedy", "random"], seed=2))
    exported = call(app, "export")
    exported["seed"] += 1
    reply = json.loads(
        app.handle(json.dumps({"type": "load_replay", "payload": {"replay": exported}}))
    )
    assert reply["ok"] is False


def test_bots_are_not_given_the_deal_seed(tmp_path):
    bot_dir = tmp_path / "spy"
    bot_dir.mkdir()
    (bot_dir / "bot.py").write_text(
        "SEEN = []\n"
        "class Spy:\n"
        "    def reset(self, seat, rules, seed):\n"
        "        SEEN.append(seed)\n"
        "        self.seed = seed\n"
        "    def act(self, view, legal):\n"
        "        return legal[0]\n"
        "def create_bot(artifact_dir):\n"
        "    return Spy()\n"
    )
    game = session.Game(Rules(n_players=2), ["spy", "spy"], 123456, tmp_path)
    assert all(bot.seed != 123456 for bot in game.bots.values())


def test_artifact_dir_is_only_given_when_files_exist(tmp_path):
    source = (
        "class Bot:\n"
        "    def __init__(self, artifact_dir):\n"
        "        self.artifact_dir = artifact_dir\n"
        "def create_bot(artifact_dir):\n"
        "    return Bot(artifact_dir)\n"
    )
    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "bot.py").write_text(source)
    assert session.load_bot(tmp_path, "plain").artifact_dir is None

    (folder / "weights.npz").write_bytes(b"x")
    assert session.load_bot(tmp_path, "plain").artifact_dir == folder


def test_lowest_card_is_played_first_in_view(app):
    deal = new_game(Rules(n_players=2), 5)
    snapshot = call(app, "new_game", rules=rules(), seats=["greedy", "greedy"], seed=5)
    first = min(min(hand) for hand in deal.hands)
    moved = call(app, "step")["view"]["moves"][0]
    assert moved["seat"] == snapshot["view"]["first_leader"]
    assert first in {parse_card(card) for card in moved["play"]["cards"]}
