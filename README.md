# rl-games

Tiến Lên engine, versioned bots, tournaments, RL training, and a browser app, all driven by one CLI.

## Setup

```bash
uv sync                # add --extra rl for training
uv run tienlen db init
```

Run every command below as `uv run tienlen ...`. Add `--json` to any command for machine-readable output. State lives in `./data`, or in `$TIENLEN_DATA` if set.

## Essential commands

```bash
# Write and register a bot
tienlen bot new mybot
tienlen bot validate bots/mybot/bot.py
tienlen bot register bots/mybot/bot.py --kind rule   # also plays 10,000 games against each other bot
tienlen bot test mybot
tienlen bot list

# Compare bots
tienlen arena run mybot greedy --games 400 --players 2
tienlen arena leaderboard
tienlen arena h2h mybot greedy

# Inspect games
tienlen match list
tienlen match replay <match_id>

# Train
tienlen train start --players 2 --opponents random@latest,greedy@latest
tienlen train list
tienlen train show <run_id>

# Grow models step by step and record what helped
tienlen lab step lab_02_mlp --change "add a hidden layer" --hypothesis "non-linear scoring wins more"
tienlen lab log
tienlen lab show 2
tienlen lab report

# Watch bots play, or play against one
tienlen web build
tienlen web serve

# Query the database
tienlen db tables
tienlen db query bot_versions --where "bot == 'greedy'"
```

Bot refs look like `greedy@3`, `greedy@latest`, or `greedy@best`. A bare name means `@latest`.

## Docs

- [Engine API](docs/index.md)
- [Registry and full CLI reference](docs/registry-cli.md)
- [RL training](docs/rl.md)
- [Model evolution](docs/evolution.md)
- [Web app](web/README.md)
