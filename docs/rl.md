# RL training

Everything here lives in `tienlen.rl`. Install the trainer's dependency with `uv sync --extra rl`. Only `tienlen.rl.trainers.ppo` imports torch, so the CLI and trained bots run without it.

## Quick start

```bash
tienlen train start --players 2 --opponents random@latest,greedy@latest \
  --iterations 130 --episodes 8000 --workers 8 --snapshot-every 20
tienlen train list
tienlen train show <run_id>
tienlen train stop <run_id>      # stops before the next iteration
tienlen train resume <run_id> --iterations 200
```

`--set key=value` overrides any trainer setting, for example `--set lr=1e-4`. Every command takes `--json`.

Each run lives in `data/runs/<run_id>/`:

| Path | Contents |
|---|---|
| `config.json` | Full run settings |
| `metrics.jsonl` | One line per iteration: losses, timing, win rate per opponent |
| `checkpoint.pt` | Model and optimizer, used by `resume` |
| `snapshots/<iteration>/weights.npz` | Exported weights |
| `status` | `running`, `finished`, `stopped`, or `failed` |

Every snapshot is registered as a new version of the bot, with `parent_version` chained and `train_run_id` set. Trained versions are ordinary bots, so `tienlen arena` and `tienlen match` work on them.

## Env

`TienLenEnv` plays one learner seat. Opponents are `TienLenBot` instances, so any registered bot can sit at the table.

```python
from tienlen.engine import Rules
from tienlen.rl.env import TienLenEnv

env = TienLenEnv(Rules(n_players=2), [opponent], seed=0)
step = env.reset()  # optional: seat=, deal_seed=, opponents=
while not step.done:
    step = env.step(step.legal[0])  # pass a Play from step.legal
print(step.reward)
```

- `Step` holds `view`, `legal`, `reward`, and `done`. The learner only sees a `PlayerView`, and the `GameState` stays private to the env.
- `reset` plays opponents until the learner's first turn, and `step` plays them until its next one. After the learner places, the env plays the game out and pays the reward at the end.
- `reward` is a function of `(place, n_players)`. `placement_reward` scales from 1 for first to -1 for last, and `win_reward` pays 1 for first only. `REWARDS` names them for `--set reward=win`.
- Opponents get a seed drawn from the env's RNG, never the deal seed.
- An opponent that returns an illegal play raises `IllegalPlay`. The env does no forfeit handling; `tienlen.arena` does.

## Encoding

`tienlen.rl.encoding` is the single source of features, shared by training and inference. It needs only numpy.

A fixed action output can't work, because one 12-card sequence alone has 4^12 suit combinations. Instead:

- `encode_state(view)` returns `STATE_DIM` floats: own hand, played cards, current play, rank counts, relative seat data, and rule flags.
- `encode_actions(view, legal)` returns one `ACTION_DIM` row per legal play: its cards, combo type, length, top card, chop flag, 2s spent, cards left, ranks left unpaired, and share of unseen cards above its top.

The network scores each row, and a softmax over the rows gives the policy.

Changing either encoder invalidates existing weights, because the input sizes or meanings change.

## Policy

`tienlen.rl.policy.Policy` runs the network with numpy from a `weights.npz`. `PolicyBot` wraps it as a `TienLenBot`, playing the top-scoring action, or sampling with `sample=True`.

`bots/neural/bot.py` is a normal bot file that loads `weights.npz` from `artifact_dir`. It needs no torch, so it can run in the browser.

## League

`League` picks each opponent for each episode:

| Share | Opponent |
|---|---|
| `self_share` (0.4) | The current policy, sampling |
| `snapshot_share` (0.3) | One of the last 10 snapshots |
| The rest | A fixed bot from `--opponents` |

Empty tiers fall back to the fixed bots, then to the current policy. Fixed opponents are given as registry refs and stored as `Opponent(name, path, artifact_dir)`, which is picklable for worker processes.

## PPO trainer

`tienlen.rl.trainers.ppo` runs this loop:

1. Sample opponents for `episodes` games and split them across `workers` processes. `workers=0` runs inline.
2. Each worker plays with the numpy `Policy`, and only the learner seat's moves are recorded.
3. Advantages use GAE with no discounting, since the only reward is at the end of the game.
4. PPO updates run for `epochs` passes over shuffled minibatches.
5. Metrics and a checkpoint are written, and a snapshot every `snapshot_every` iterations.

Settings are the fields of `PPOConfig`.

## Writing a new trainer

1. Add `src/tienlen/rl/trainers/<name>.py` defining:

   ```python
   def train(
       settings: dict,
       run: RunDir,
       opponents: Sequence[Opponent],
       on_snapshot: Callable[[int, Path], None] = ...,
   ) -> str: ...
   ```

   `settings` is the run's `config.json`. Return `"finished"` or `"stopped"`. Check `run.stop_requested()` between iterations.
2. Add `<name>` to `TRAINERS` in `rl/trainers/__init__.py`. Import heavy dependencies inside the trainer module only.
3. Write metrics with `run.log_metrics`, including `iteration` and an `opponents` map if you want `train show` to print win rates.
4. Save a snapshot directory the bot file can load, then call `on_snapshot(iteration, path)` to register it.

To train a different architecture, give it its own bot folder and pass `--bot <folder>`. The bot file decides how to read the snapshot.
