import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Annotated, Any

import typer

JsonOption = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]
Row = Mapping[str, Any]


def emit_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


def emit_rows(rows: Sequence[Row], columns: Sequence[str], as_json: bool) -> None:
    if as_json:
        emit_json(list(rows))
    elif not rows:
        typer.echo("(none)")
    else:
        typer.echo(_table(rows, columns))


def emit_record(record: Row, as_json: bool) -> None:
    if as_json:
        emit_json(record)
        return
    width = max(len(key) for key in record)
    for key, value in record.items():
        typer.echo(f"{key:<{width}}  {_cell(value)}")


def _table(rows: Iterable[Row], columns: Sequence[str]) -> str:
    cells = [[_cell(row.get(column)) for column in columns] for row in rows]
    widths = [max(len(line) for line in lines) for lines in zip(columns, *cells)]
    lines = [columns, *cells]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines
    )


def _cell(value: Any) -> str:
    match value:
        case None:
            return "-"
        case float():
            return f"{value:.3f}"
        case datetime():
            return value.strftime("%Y-%m-%d %H:%M:%S")
        case list() | tuple():
            return " ".join(_cell(item) for item in value)
        case _:
            return str(value)
