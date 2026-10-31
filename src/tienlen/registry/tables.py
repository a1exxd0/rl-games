import fcntl
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.expressions import AlwaysTrue, BooleanExpression, EqualTo, Reference
from pyiceberg.expressions.literals import literal
from pyiceberg.table import Table
from pyparsing import ParseException

from tienlen.errors import RegistryError
from tienlen.registry.schemas import NAMESPACE, TABLES

Row = dict[str, Any]
Where = BooleanExpression | str
NO_MATCH_WARNING = "Delete operation did not match any records"


def column_is[V: (str, int)](column: str, value: V) -> EqualTo:
    return EqualTo(term=Reference(column), value=literal(value))


class TableStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock_path = root / ".write.lock"
        warehouse = root / "warehouse"
        warehouse.mkdir(parents=True, exist_ok=True)
        self.catalog = SqlCatalog(
            "tienlen",
            uri=f"sqlite:///{root}/catalog.db",
            warehouse=f"file://{warehouse}",
        )

    @contextmanager
    def lock(self) -> Iterator[None]:
        with self._lock_path.open("w") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            yield

    def ensure_tables(self) -> None:
        with self.lock():
            self.catalog.create_namespace_if_not_exists(NAMESPACE)
            for name, schema in TABLES.items():
                self.catalog.create_table_if_not_exists(f"{NAMESPACE}.{name}", schema=schema)

    def migrate(self) -> None:
        with self.lock():
            for name, schema in TABLES.items():
                with self.table(name).update_schema() as update:
                    update.union_by_name(schema)

    def table(self, name: str) -> Table:
        try:
            return self.catalog.load_table(f"{NAMESPACE}.{name}")
        except NoSuchTableError as error:
            raise RegistryError(f"unknown table: {name}") from error

    def read(
        self,
        name: str,
        where: Where | None = None,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[Row]:
        table = self.table(name)
        try:
            scan = table.scan(
                row_filter=where if where is not None else AlwaysTrue(),
                selected_fields=tuple(columns or ("*",)),
                limit=limit,
            )
            return scan.to_arrow().to_pylist()
        except (ParseException, ValueError) as error:
            raise RegistryError(f"invalid query on {name}: {where or columns}") from error

    def append(self, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        self.table(name).append(pa.Table.from_pylist(list(rows), schema=TABLES[name]))

    def replace(self, name: str, where: Where, rows: Sequence[Mapping[str, Any]]) -> None:
        arrow = pa.Table.from_pylist(list(rows), schema=TABLES[name])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=NO_MATCH_WARNING)
            self.table(name).overwrite(arrow, overwrite_filter=where)

    def summary(self) -> list[Row]:
        return [self._summarize(name) for name in TABLES]

    def snapshots(self, name: str) -> list[Row]:
        columns = ("committed_at", "snapshot_id", "parent_id", "operation")
        return self.table(name).inspect.snapshots().select(columns).to_pylist()

    def compact(self, name: str) -> Row:
        with self.lock():
            before = len(self.table(name).inspect.files())
            rows = self.table(name).scan().to_arrow()
            if rows.num_rows:
                self.replace(name, AlwaysTrue(), rows.to_pylist())
                self.table(name).maintenance.expire_snapshots().older_than(now()).commit()
            after = len(self.table(name).inspect.files())
        return {"table": name, "rows": rows.num_rows, "files_before": before, "files_after": after}

    def _summarize(self, name: str) -> Row:
        table = self.table(name)
        return {
            "table": name,
            "rows": table.scan().count(),
            "files": len(table.inspect.files()),
            "snapshots": len(table.snapshots()),
        }


def now() -> datetime:
    return datetime.now(UTC)
