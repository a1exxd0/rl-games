# Registry and CLI

All state lives under `$TIENLEN_DATA`, default `./data`:

```text
catalog.db    Iceberg SQL catalog (SQLite)
warehouse/    Iceberg tables
code/<sha>/   immutable bot.py copies
artifacts/<sha>/  immutable weights and other files
replays/<arena or "matches">/<match_id>.json
```

Every command takes `--json`. Errors go to stderr with exit code 1.

## Bot refs

`name@N`, `name@latest` (or plain `name`), `name@best`. `latest` and `best` skip retired versions. `best` is the highest `mu - 3 * sigma`; pass `rules=` to `Registry.resolve`, otherwise it uses the rule set the bot has played most.

## Commands

| Command | Does |
|---|---|
| `bot new NAME [--dir bots]` | Write `bots/NAME/bot.py` from the template |
| `bot validate PATH\|REF` | Load it and play 9 games across 2-4 seats; exit 1 on problems |
| `bot register PATH [--name --kind --description --artifact-dir --parent --train-run --rule --no-check --no-test --test-games --workers]` | Store a new version, then run `bot test` on it. Identical code, artifacts, and rules return the existing version |
| `bot test REF [--games --seed --rule --workers --move-timeout]` | Win rate against every other bot, 10,000 2-player games per opponent by default |
| `bot list`, `bot versions NAME`, `bot show REF` | Inspect versions, dependencies, ratings |
| `bot retire REF` | Exclude from `latest` and `best`; stays addressable as `name@N` |
| `bot export REF --out DIR` | Copy `bot.py` and `artifacts/` out |
| `match run REF... [--games --seed --players --rule --move-timeout]` | Fixed seats, one ref per seat, recorded with no ratings update |
| `match list [--arena --bot --limit]`, `match show ID` | Inspect; ids accept prefixes |
| `match replay ID` | Verify against the engine, then print the deal and moves |
| `arena run REF... [--games --seed --players --rule --workers --move-timeout --batch-size]` | Rotated seats, recorded in batches, updates ratings |
| `arena leaderboard [--players --rule --limit]` | Ratings, best first, per rule set |
| `arena h2h A B` | First ref's record over recorded games, with a 95% Wilson interval |
| `train ...`, `web ...` | See `tienlen train --help` and `tienlen web --help` |
| `db init` | Create tables and add new columns from `schemas.py` |
| `db tables`, `db snapshots TABLE` | Row, file, and snapshot counts |
| `db query TABLE [--where --columns --limit]` | Read rows; `--where` is Iceberg filter syntax such as `bot == 'greedy'` |
| `db compact [TABLE]` | Rewrite into fewer files and expire old snapshots |
| `dev rules [--players --rule]` | Print a config and its `rules_id` |
| `dev bench [--games --players --rule --seed]` | Engine throughput |

`bot test` plays the latest active version of each other bot, or the version a lab step recorded, plus the latest other version of the bot itself. Pairs already holding enough recorded games are not replayed, so a rerun only tops up. Games go into `matches` and update ratings, but keep no replay files. `lab step` runs the same test on its best seed; pass `--no-test` to skip it. `Registry.register` never runs it, so training snapshots stay cheap.

`--rule` is `key=value`, repeatable, using `Rules` field names: `--rule lock_on_pass=false`.

## Tables

Namespace `tienlen`. Schemas live in `src/tienlen/registry/schemas.py`. To add a column, add it there and run `db init`.

| Table | Fields |
|---|---|
| `bots` | bot, kind, description, created_at |
| `bot_versions` | bot, version, kind, code_sha, artifact_sha, parent_version, train_run_id, rules_id, engine_version, browser_compatible, status, created_at |
| `matches` | match_id, arena_id, seed, rules_id, engine_version, seats, placements, forfeits, turns, replay_uri, played_at |
| `ratings` | bot, version, rules_id, n_players, mu, sigma, games, computed_at |
| `train_runs` | run_id, bot, config, status, metrics_uri, started_at, finished_at |

`seats` and `placements` are `name@N` refs and seat numbers. `placements` is finishing order, with forfeits last. `replay_uri` is relative to the data directory.

## Python API

```python
from tienlen.arena import Entrant, MatchResult, play_match, run_arena
from tienlen.bots.loader import load_bot
from tienlen.registry import Registry

registry = Registry.open()
version = registry.resolve("greedy@latest")
bot = load_bot(version.path, version.artifact_dir)

results = list(run_arena([version.entrant, other.entrant], rules, games=100, seed=1, workers=4))
registry.record_matches(results, arena_id="my-arena")
registry.update_ratings(rules.rules_id, rules.n_players)
```

`record_matches` does not update ratings. Seats that are not `name@N` refs are stored but skipped by ratings.

## Behaviour

- **Writes:** each write takes a file lock on `data/.write.lock`, so separate processes can share a data directory. Commit in batches: one `record_matches` call is one Iceberg commit.
- **Forfeits:** a crash, illegal move, or timeout forfeits. The seat keeps playing a fallback (pass, else lowest play) so the game stays replayable, and it places last.
- **Timeouts:** `SIGALRM` on the main thread, which is what arena workers use. Off the main thread, or in C code that never yields, a slow move only forfeits after it returns.
- **Ratings:** PlackettLuce from OpenSkill, recomputed from `matches` per `rules_id` and seat count. Bots are immutable, so game order carries no signal, yet sequential updates depend on it heavily. Ratings therefore average up to 32 seeded shuffles. Earlier ratings stay in Iceberg snapshots until `db compact`.
- **Self-play:** games where every seat is the same ref are not rated.
- **Seat rotation:** each deal is played once per rotation, so `--games` should be a multiple of the seat count.
- **`h2h`:** the interval treats games as independent, but rotated games share deals. In 3-4 seat games, a win means a better mean finishing position.
- **Seeds:** bots get `sha256(deal seed, seat)`, never the deal seed. Arena deal seeds are 63-bit so they can't be guessed from a hand.
