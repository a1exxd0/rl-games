import json
import os
import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pyiceberg.expressions import And

from tienlen.arena import Entrant, MatchResult, Rating, compute_ratings, head_to_head
from tienlen.engine import ENGINE_VERSION, Rules, moves_to_list
from tienlen.errors import RegistryError
from tienlen.registry.artifacts import (
    BOT_FILE,
    artifact_path,
    code_path,
    store_artifacts,
    store_code,
)
from tienlen.registry.metadata import browser_compatible
from tienlen.registry.tables import Row, TableStore, column_is, now

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
ACTIVE = "active"
RETIRED = "retired"
LATEST = "latest"
BEST = "best"


@dataclass(frozen=True)
class BotVersion:
    name: str
    version: int
    path: Path
    artifact_dir: Path | None
    code_sha: str
    kind: str
    rules_id: str
    browser_compatible: bool

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def entrant(self) -> Entrant:
        return Entrant(self.ref, self.path, self.artifact_dir)


class Registry:
    def __init__(self, root: Path, tables: TableStore) -> None:
        self.root = root
        self.tables = tables

    @classmethod
    def open(cls, data_dir: Path | None = None) -> Self:
        root = Path(data_dir or os.environ.get("TIENLEN_DATA", "data")).resolve()
        root.mkdir(parents=True, exist_ok=True)
        tables = TableStore(root)
        tables.ensure_tables()
        return cls(root, tables)

    def resolve(self, ref: str, *, rules: Rules | None = None) -> BotVersion:
        name, _, selector = ref.partition("@")
        rows = self._version_rows(name)
        if not rows:
            raise RegistryError(f"unknown bot: {name}")
        return self._to_version(self._select(rows, selector or LATEST, ref, rules))

    def versions(self, name: str | None = None) -> list[BotVersion]:
        rows = self.version_rows(name)
        return [self._to_version(row) for row in rows]

    def version_rows(self, name: str | None = None) -> list[Row]:
        rows = self._version_rows(name) if name else self.tables.read("bot_versions")
        return sorted(rows, key=lambda row: (row["bot"], row["version"]))

    def register(
        self,
        path: Path,
        *,
        kind: str,
        artifact_dir: Path | None = None,
        parent: str | None = None,
        train_run_id: str | None = None,
        rules: Rules | None = None,
        name: str | None = None,
        description: str = "",
    ) -> BotVersion:
        path = Path(path)
        name = name or _default_name(path)
        if not NAME_PATTERN.fullmatch(name):
            raise RegistryError(f"invalid bot name: {name}")
        if not path.is_file():
            raise RegistryError(f"no such file: {path}")
        with self.tables.lock():
            row = {
                "kind": kind,
                "code_sha": store_code(self.root, path),
                "artifact_sha": store_artifacts(self.root, Path(artifact_dir))
                if artifact_dir
                else None,
                "parent_version": self.resolve(parent).ref if parent else None,
                "train_run_id": train_run_id,
                "rules_id": (rules or Rules()).rules_id,
                "browser_compatible": browser_compatible(path),
            }
            return self._to_version(self._add_version(name, description, row))

    def retire(self, ref: str) -> BotVersion:
        version = self.resolve(ref)
        with self.tables.lock():
            row = next(
                row for row in self._version_rows(version.name) if row["version"] == version.version
            )
            where = And(column_is("bot", version.name), column_is("version", version.version))
            self.tables.replace("bot_versions", where, [{**row, "status": RETIRED}])
        return version

    def start_run(self, bot: str, config: dict) -> str:
        run_id = uuid.uuid4().hex
        row = {
            "run_id": run_id,
            "bot": bot,
            "config": json.dumps(config, sort_keys=True, default=str),
            "status": "running",
            "started_at": now(),
        }
        with self.tables.lock():
            self.tables.append("train_runs", [row])
        return run_id

    def finish_run(self, run_id: str, status: str, metrics_uri: str | None = None) -> None:
        where = column_is("run_id", run_id)
        with self.tables.lock():
            rows = self.tables.read("train_runs", where)
            if not rows:
                raise RegistryError(f"unknown run: {run_id}")
            finished = {
                **rows[0],
                "status": status,
                "metrics_uri": metrics_uri,
                "finished_at": now(),
            }
            self.tables.replace("train_runs", where, [finished])

    def runs(self, run_id: str | None = None) -> list[Row]:
        where = column_is("run_id", run_id) if run_id else None
        rows = self.tables.read("train_runs", where)
        return sorted(rows, key=lambda row: row["started_at"], reverse=True)

    def start_experiment(
        self, *, step: int, parent_ref: str | None, change: str, hypothesis: str, config: dict
    ) -> str:
        experiment_id = uuid.uuid4().hex
        row = {
            "experiment_id": experiment_id,
            "step": step,
            "parent_ref": parent_ref,
            "change": change,
            "hypothesis": hypothesis,
            "config": json.dumps(config, sort_keys=True),
            "verdict": "pending",
            "created_at": now(),
        }
        with self.tables.lock():
            self.tables.append("experiments", [row])
        return experiment_id

    def update_experiment(self, experiment_id: str, **fields: Any) -> Row:
        where = column_is("experiment_id", experiment_id)
        with self.tables.lock():
            rows = self.tables.read("experiments", where)
            if not rows:
                raise RegistryError(f"unknown experiment: {experiment_id}")
            updated = {**rows[0], **fields}
            self.tables.replace("experiments", where, [updated])
        return updated

    def put_reference(self, experiment_id: str, **fields: Any) -> Row:
        row = {
            "experiment_id": experiment_id,
            "step": 0,
            "created_at": now(),
            **fields,
        }
        with self.tables.lock():
            self.tables.replace("experiments", column_is("experiment_id", experiment_id), [row])
        return row

    def experiments(self) -> list[Row]:
        return sorted(self.tables.read("experiments"), key=lambda row: row["step"])

    def experiment(self, step: int) -> Row:
        found = [row for row in self.experiments() if row["step"] == step]
        if not found:
            raise RegistryError(f"unknown step: {step}")
        return found[0]

    def record_matches(
        self, results: Sequence[MatchResult], *, arena_id: str | None = None, replays: bool = True
    ) -> list[str]:
        rows = [self._match_row(result, arena_id, replays) for result in results]
        if rows:
            with self.tables.lock():
                self.tables.append("matches", rows)
        return [row["match_id"] for row in rows]

    def matches(
        self,
        *,
        arena_id: str | None = None,
        bot: str | None = None,
        limit: int | None = None,
    ) -> list[Row]:
        where = column_is("arena_id", arena_id) if arena_id else None
        rows = sorted(
            self.tables.read("matches", where),
            key=lambda row: row["played_at"],
            reverse=True,
        )
        if bot:
            rows = [row for row in rows if any(_same_bot(seat, bot) for seat in row["seats"])]
        return rows[:limit]

    def match(self, match_id: str) -> Row:
        found = [
            row for row in self.tables.read("matches", f"match_id LIKE '{_id_prefix(match_id)}%'")
        ]
        if len(found) != 1:
            raise RegistryError(f"{len(found)} matches for id {match_id}")
        return found[0]

    def replay(self, match_id: str) -> dict[str, Any]:
        uri = self.match(match_id)["replay_uri"]
        if uri is None:
            raise RegistryError(f"no replay stored for {match_id}")
        return json.loads((self.root / uri).read_text())

    def update_ratings(
        self, rules_id: str | None = None, n_players: int | None = None
    ) -> list[Row]:
        with self.tables.lock():
            groups = self._games_by_rules(rules_id, n_players)
            rows = [
                row
                for (group_rules, players), games in groups.items()
                for row in self._replace_ratings(group_rules, players, games)
            ]
        return rows

    def leaderboard(self, rules_id: str | None = None, n_players: int | None = None) -> list[Row]:
        rows = [
            {**row, "ordinal": row["mu"] - 3 * row["sigma"]}
            for row in self.tables.read("ratings", _ratings_filter(rules_id, n_players))
        ]
        return sorted(
            rows,
            key=lambda row: (row["rules_id"], row["n_players"], -row["ordinal"]),
        )

    def head_to_head(self, first: str, second: str) -> list[Row]:
        first, second = self.resolve(first).ref, self.resolve(second).ref
        return [
            _h2h_row(group_rules, players, games, first, second)
            for (group_rules, players), games in self._games_by_rules(None, None).items()
            if any(first in seats and second in seats for seats, _ in games)
        ]

    def win_rates(self, rules: Rules) -> list[Row]:
        key = (rules.rules_id, rules.n_players)
        by_pair: dict[tuple[str, ...], list[tuple[list[str], list[int]]]] = {}
        for seats, order in self._games_by_rules(*key).get(key, []):
            by_pair.setdefault(tuple(sorted(set(seats))), []).append((seats, order))
        return [
            _h2h_row(*key, games, first, second)
            for pair, games in by_pair.items()
            if len(pair) == 2
            for first, second in (pair, pair[::-1])
        ]

    def _version_rows(self, name: str) -> list[Row]:
        return self.tables.read("bot_versions", column_is("bot", name))

    def _to_version(self, row: Row) -> BotVersion:
        artifact_sha = row["artifact_sha"]
        return BotVersion(
            name=row["bot"],
            version=row["version"],
            path=code_path(self.root, row["code_sha"]),
            artifact_dir=artifact_path(self.root, artifact_sha) if artifact_sha else None,
            code_sha=row["code_sha"],
            kind=row["kind"],
            rules_id=row["rules_id"],
            browser_compatible=row["browser_compatible"],
        )

    def _select(self, rows: Sequence[Row], selector: str, ref: str, rules: Rules | None) -> Row:
        if selector.isdecimal():
            found = [row for row in rows if row["version"] == int(selector)]
        elif selector == LATEST:
            found = [max(_active(rows, ref), key=lambda row: row["version"])]
        elif selector == BEST:
            found = [self._best(rows, ref, rules)]
        else:
            raise RegistryError(f"invalid bot ref: {ref}")
        if not found:
            raise RegistryError(f"unknown version: {ref}")
        return found[0]

    def _best(self, rows: Sequence[Row], ref: str, rules: Rules | None) -> Row:
        active = {row["version"]: row for row in _active(rows, ref)}
        ratings = [
            rating
            for rating in self.tables.read("ratings", column_is("bot", rows[0]["bot"]))
            if rating["version"] in active
        ]
        ratings = _ratings_for(ratings, rules)
        if not ratings:
            raise RegistryError(f"no ratings for {ref}")
        return active[max(ratings, key=_ordinal)["version"]]

    def _add_version(self, name: str, description: str, row: Row) -> Row:
        rows = self._version_rows(name)
        same = _matching_version(rows, row)
        if same is not None:
            return same
        created_at = now()
        if not rows:
            bot = {
                "bot": name,
                "kind": row["kind"],
                "description": description,
                "created_at": created_at,
            }
            self.tables.append("bots", [bot])
        added = {
            **row,
            "bot": name,
            "version": max((old["version"] for old in rows), default=0) + 1,
            "engine_version": ENGINE_VERSION,
            "status": ACTIVE,
            "created_at": created_at,
        }
        self.tables.append("bot_versions", [added])
        return added

    def _match_row(self, result: MatchResult, arena_id: str | None, replays: bool) -> Row:
        match_id = uuid.uuid4().hex
        return {
            "match_id": match_id,
            "arena_id": arena_id,
            "seed": result.seed,
            "rules_id": result.rules_id,
            "engine_version": ENGINE_VERSION,
            "seats": list(result.seats),
            "placements": list(result.placements),
            "forfeits": list(result.forfeits),
            "turns": len(result.moves),
            "replay_uri": self._write_replay(result, arena_id, match_id) if replays else None,
            "played_at": now(),
        }

    def _write_replay(self, result: MatchResult, arena_id: str | None, match_id: str) -> str | None:
        if result.rules is None:
            return None
        relative = Path("replays") / (arena_id or "matches") / f"{match_id}.json"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rules": result.rules.to_dict(),
            "seed": result.seed,
            "seats": list(result.seats),
            "placements": list(result.placements),
            "forfeits": list(result.forfeits),
            "moves": moves_to_list(result.moves),
        }
        target.write_text(json.dumps(payload))
        return relative.as_posix()

    def _games_by_rules(
        self, rules_id: str | None, n_players: int | None
    ) -> dict[tuple[str, int], list[tuple[list[str], list[int]]]]:
        where = column_is("rules_id", rules_id) if rules_id else None
        rows = self.tables.read(
            "matches", where, columns=("rules_id", "seats", "placements", "played_at", "match_id")
        )
        rows.sort(key=lambda row: (row["played_at"], row["match_id"]))
        groups: dict[tuple[str, int], list[tuple[list[str], list[int]]]] = {}
        for row in rows:
            players = len(row["seats"])
            if n_players in (None, players):
                groups.setdefault((row["rules_id"], players), []).append(
                    (row["seats"], row["placements"])
                )
        return groups

    def _replace_ratings(
        self, rules_id: str, n_players: int, games: Iterable[tuple[list[str], list[int]]]
    ) -> list[Row]:
        computed_at = now()
        rows = [
            _rating_row(rating, rules_id, n_players, computed_at)
            for rating in compute_ratings(games)
        ]
        self.tables.replace("ratings", _ratings_filter(rules_id, n_players), rows)
        return rows


def _default_name(path: Path) -> str:
    return path.parent.resolve().name if path.name == BOT_FILE else path.stem


def _active(rows: Sequence[Row], ref: str) -> list[Row]:
    active = [row for row in rows if row["status"] == ACTIVE]
    if not active:
        raise RegistryError(f"no active versions for {ref}")
    return active


def _matching_version(rows: Sequence[Row], row: Row) -> Row | None:
    identity = ("code_sha", "artifact_sha", "rules_id")
    return next(
        (old for old in rows if all(old[field] == row[field] for field in identity)),
        None,
    )


def _ordinal(row: Row) -> float:
    return row["mu"] - 3 * row["sigma"]


def _ratings_for(ratings: list[Row], rules: Rules | None) -> list[Row]:
    if rules is not None:
        return [row for row in ratings if row["rules_id"] == rules.rules_id]
    totals: dict[str, int] = {}
    for row in ratings:
        totals[row["rules_id"]] = totals.get(row["rules_id"], 0) + row["games"]
    if not totals:
        return []
    busiest = max(totals, key=totals.__getitem__)
    return [row for row in ratings if row["rules_id"] == busiest]


def _ratings_filter(rules_id: str | None, n_players: int | None) -> Any:
    terms = []
    if rules_id:
        terms.append(column_is("rules_id", rules_id))
    if n_players:
        terms.append(column_is("n_players", n_players))
    if len(terms) == 2:
        return And(*terms)
    return terms[0] if terms else None


def _rating_row(rating: Rating, rules_id: str, n_players: int, computed_at: Any) -> Row:
    return {
        "bot": rating.bot,
        "version": rating.version,
        "rules_id": rules_id,
        "n_players": n_players,
        "mu": rating.mu,
        "sigma": rating.sigma,
        "games": rating.games,
        "computed_at": computed_at,
    }


def _h2h_row(rules_id: str, n_players: int, games: Iterable, first: str, second: str) -> Row:
    result = head_to_head(games, first, second)
    low, high = result.interval
    return {
        "first": first,
        "second": second,
        "rules_id": rules_id,
        "n_players": n_players,
        "games": result.wins + result.losses + result.ties,
        "wins": result.wins,
        "losses": result.losses,
        "ties": result.ties,
        "win_rate": result.rate,
        "ci_low": low,
        "ci_high": high,
    }


def _same_bot(seat: str, bot: str) -> bool:
    return seat == bot or seat.partition("@")[0] == bot


def _id_prefix(match_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]+", match_id):
        raise RegistryError(f"invalid match id: {match_id}")
    return match_id
