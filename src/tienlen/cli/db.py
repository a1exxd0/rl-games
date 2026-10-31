from typing import Annotated

import typer

from tienlen.cli.options import open_registry
from tienlen.cli.output import JsonOption, emit_rows

app = typer.Typer(help="Inspect and maintain the Iceberg tables.", no_args_is_help=True)

SUMMARY_COLUMNS = ("table", "rows", "files", "snapshots")
SNAPSHOT_COLUMNS = ("committed_at", "snapshot_id", "parent_id", "operation")
COMPACT_COLUMNS = ("table", "rows", "files_before", "files_after")


@app.command(help="Create missing tables and add new schema columns.")
def init(json_output: JsonOption = False) -> None:
    registry = open_registry()
    registry.tables.migrate()
    emit_rows(registry.tables.summary(), SUMMARY_COLUMNS, json_output)


@app.command(help="List tables with row, file, and snapshot counts.")
def tables(json_output: JsonOption = False) -> None:
    emit_rows(open_registry().tables.summary(), SUMMARY_COLUMNS, json_output)


@app.command(help="Read rows from a table.")
def query(
    table: str,
    where: Annotated[
        str | None, typer.Option("--where", help="Filter, e.g. \"bot == 'greedy'\".")
    ] = None,
    columns: Annotated[
        str | None, typer.Option("--columns", help="Comma-separated column names.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Rows to read.")] = 20,
    json_output: JsonOption = False,
) -> None:
    selected = columns.split(",") if columns else None
    rows = open_registry().tables.read(table, where, selected, limit)
    emit_rows(rows, selected or list(rows[0]) if rows else (), json_output)


@app.command(help="List a table's snapshots.")
def snapshots(table: str, json_output: JsonOption = False) -> None:
    emit_rows(open_registry().tables.snapshots(table), SNAPSHOT_COLUMNS, json_output)


@app.command(help="Rewrite a table, or all, into one file set and expire old snapshots.")
def compact(
    table: Annotated[str | None, typer.Argument(help="Table name. Defaults to all.")] = None,
    json_output: JsonOption = False,
) -> None:
    store = open_registry().tables
    names = [table] if table else [row["table"] for row in store.summary()]
    emit_rows([store.compact(name) for name in names], COMPACT_COLUMNS, json_output)
