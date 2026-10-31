# Web app

Static site running the Python engine in Pyodide. All rules live in `tienlen.engine`; the JS only renders snapshots and picks from legal plays.

| File | Job |
|---|---|
| `py/session.py` | Games, replays, and the JSON command dispatcher, inside Pyodide |
| `js/protocol.js` | Worker message client and the pinned Pyodide URL |
| `js/worker.js` | Boots Pyodide, installs the wheel, fetches bots on first use |
| `js/game.js` | Game control: modes, auto-play loop, stale reply handling |
| `js/layout.js` | Pure table geometry: where every card and plate rests |
| `js/scene.js` | Stable card and plate elements, and the transitions between snapshots |
| `js/motion.js` | Speed timings and Web Animations helpers |
| `js/cards.js` | SVG card faces and backs |
| `js/render.js` | Moves log and status text |
| `js/models.js` | Models tab, drawn from `models.json` |
| `js/grid.js` | Win-rate grid on the Models tab |
| `js/main.js` | Page wiring: setup form, controls, card selection |

## Run

```sh
uv run tienlen web build
uv run tienlen web serve
```

The site builds to `web/dist/`. `serve` rebuilds first unless you pass `--no-rebuild`.

Watch and human games start playing as soon as they open. Replays stay paused until you press Auto-play. Space pauses and resumes auto-play in watch and replay modes.

## Bots and models

`build` reads the registry at `$TIENLEN_DATA` (default `data/`) without writing to it, so a rebuild is all a new model needs.

- Each browser-compatible registered version goes into `dist/bots/<ref>/` with its code and weights.
- `bots/<name>/` folders are added when no registered version has that name, the bot imports only what the browser has, and `create_bot` succeeds.
- `models.json` holds the experiments table, in the shape of `tienlen lab log --json` plus hypothesis, notes, and whether the bot can run.

## Add a command

1. Add a method on `App` in `py/session.py` and list it in `commands`.
2. Call it with `engine.call("name", payload)`.
