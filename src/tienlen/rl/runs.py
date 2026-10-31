import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_ENV = "TIENLEN_DATA"


def data_root() -> Path:
    return Path(os.environ.get(DATA_ENV, "data"))


def runs_root(root: Path | None = None) -> Path:
    return (root or data_root()) / "runs"


@dataclass(frozen=True, slots=True)
class RunDir:
    path: Path

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def checkpoint_path(self) -> Path:
        return self.path / "checkpoint.pt"

    @property
    def status_path(self) -> Path:
        return self.path / "status"

    @property
    def stop_path(self) -> Path:
        return self.path / "stop"

    @property
    def snapshots_path(self) -> Path:
        return self.path / "snapshots"

    @property
    def parent_path(self) -> Path:
        return self.path / "parent"

    def parent(self) -> str | None:
        return self.parent_path.read_text() if self.parent_path.exists() else None

    def set_parent(self, ref: str) -> None:
        self.parent_path.write_text(ref)

    def create(self, config: dict[str, Any]) -> None:
        self.path.mkdir(parents=True)
        self.config_path.write_text(json.dumps(config, indent=2))
        self.set_status("running")

    def config(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text())

    def status(self) -> str:
        return self.status_path.read_text()

    def set_status(self, status: str) -> None:
        self.status_path.write_text(status)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("a") as file:
            file.write(json.dumps(metrics) + "\n")

    def metrics(self) -> list[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        return [json.loads(line) for line in self.metrics_path.read_text().splitlines()]

    def request_stop(self) -> None:
        self.stop_path.touch()

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def clear_stop(self) -> None:
        self.stop_path.unlink(missing_ok=True)

    def snapshot_dir(self, iteration: int) -> Path:
        return self.snapshots_path / f"{iteration:06d}"

    def snapshots(self) -> list[Path]:
        if not self.snapshots_path.exists():
            return []
        return sorted(self.snapshots_path.iterdir())


def iter_runs(root: Path | None = None) -> Iterator[RunDir]:
    base = runs_root(root)
    if base.exists():
        yield from (RunDir(path) for path in sorted(base.iterdir()) if path.is_dir())
