# Rules

- After finishing an MR, you must run:

```
uv run pytest
uv run complexipy
uv run ruff check
uv run basedpyright
uv run ruff format
```

- Never push to remote yourself.
