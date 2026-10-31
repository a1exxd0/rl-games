import subprocess
import sys
from dataclasses import replace

import pytest

from tienlen.engine import Rules


def test_defaults():
    rules = Rules()
    assert rules.n_players == 4
    assert rules.lock_on_pass
    assert not rules.chop_after_pass
    assert not rules.instant_wins
    assert rules.first_lead_includes_lowest
    assert not rules.forbid_finish_on_two


@pytest.mark.parametrize("n_players", [0, 1, 5])
def test_player_count_is_validated(n_players):
    with pytest.raises(ValueError):
        Rules(n_players=n_players)


def test_rules_id_is_stable():
    assert Rules().rules_id == Rules().rules_id
    assert Rules().rules_id == "fcbdd735f55b"


@pytest.mark.parametrize(
    "change",
    [
        {"n_players": 2},
        {"lock_on_pass": False},
        {"chop_after_pass": True},
        {"instant_wins": True},
        {"first_lead_includes_lowest": False},
        {"forbid_finish_on_two": True},
    ],
)
def test_every_toggle_changes_the_rules_id(change):
    assert replace(Rules(), **change).rules_id != Rules().rules_id


def test_dict_round_trip():
    rules = Rules(n_players=3, instant_wins=True)
    assert Rules.from_dict(rules.to_dict()) == rules


def test_engine_avoids_third_party_imports():
    code = "import sys, tienlen.engine; print(*sys.modules)"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = {name.split(".")[0] for name in run.stdout.split()}
    assert not loaded & {"numpy", "openskill", "pyarrow", "pyiceberg", "torch", "typer"}
