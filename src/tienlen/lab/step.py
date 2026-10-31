import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from tienlen.arena import Entrant
from tienlen.errors import RegistryError
from tienlen.lab.suite import (
    BEST_ROLE,
    GREEDY_ROLE,
    RANDOM_ROLE,
    SUITE_RULES,
    Pairing,
    run_suite,
)
from tienlen.registry import Registry
from tienlen.rl.league import from_refs
from tienlen.rl.runs import RunDir, runs_root

SEEDS = (11, 22, 33)
BOT_KIND = "lab"
PROMOTED = "promoted"
REJECTED = "rejected"
PENDING = "pending"
FAILED = "failed"
REFERENCE = "reference"
REFERENCE_ID = "reference"
FOLDER_PATTERN = re.compile(r"lab_(?P<step>\d+)_[a-z0-9_]+")
SELECTION_ROLES = (RANDOM_ROLE, GREEDY_ROLE)


@dataclass(frozen=True)
class StepRequest:
    folder: str
    change: str
    hypothesis: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    budget_change: bool = False
    parent: str | None = None
    workers: int = 8
    seeds: Sequence[int] = SEEDS
    games: int | None = None


def step_number(folder: str) -> int:
    match = FOLDER_PATTERN.fullmatch(folder)
    if match is None:
        raise RegistryError(f"invalid step folder: {folder}")
    return int(match["step"])


def current_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    promoted = [row for row in rows if row["verdict"] == PROMOTED]
    return max(promoted, key=lambda row: row["step"], default=None)


def _parent_row(registry: Registry, parent: str | None) -> Mapping[str, Any] | None:
    rows = registry.experiments()
    if parent is None:
        return current_best(rows)
    found = [row for row in rows if row["bot_ref"] == parent]
    if not found:
        raise RegistryError(f"no experiment for {parent}")
    return found[0]


def merged_config(
    parent: Mapping[str, Any] | None, overrides: Mapping[str, Any], budget_change: bool
) -> dict[str, Any]:
    from tienlen.lab.trainer import BUDGET_KEYS, LabConfig

    inherited = json.loads(parent["config"]) if parent else {}
    defaults = LabConfig.from_dict(inherited).to_dict()
    changed = [key for key in BUDGET_KEYS if key in overrides and overrides[key] != defaults[key]]
    if changed and not budget_change:
        raise RegistryError(f"budget changed without --budget-change: {changed}")
    return LabConfig.from_dict({**defaults, **overrides}).to_dict()


def _train_seed(
    registry: Registry,
    request: StepRequest,
    config: dict[str, Any],
    seed: int,
    parent_ref: str | None,
    run_map,
) -> dict[str, Any]:
    from tienlen.lab.trainer import LabConfig, SeedJob, train_seed

    bot_path = (Path("bots") / request.folder / "bot.py").resolve()
    run_id = registry.start_run(request.folder, {**config, "seed": seed})
    run = RunDir(runs_root(registry.root) / run_id)
    run.create({"bot": request.folder, "seed": seed, **config})
    settings = LabConfig.from_dict(config)
    fixed = from_refs(settings.opponents, registry.resolve)
    job = SeedJob(bot_path, settings, SUITE_RULES, seed, run, fixed, request.workers)
    try:
        weights = train_seed(job, run_map)
    except BaseException:
        run.set_status(FAILED)
        registry.finish_run(run_id, FAILED, metrics_uri=str(run.metrics_path))
        raise
    run.set_status("finished")
    registry.finish_run(run_id, "finished", metrics_uri=str(run.metrics_path))
    version = registry.register(
        bot_path,
        kind=BOT_KIND,
        artifact_dir=weights,
        train_run_id=run_id,
        rules=SUITE_RULES,
        name=request.folder,
        parent=parent_ref,
    )
    return {"ref": version.ref, "seed": seed, "run_id": run_id}


def _train_all(
    registry: Registry, request: StepRequest, config: dict[str, Any], parent_ref: str | None
) -> tuple[list[dict[str, Any]], float]:
    from tienlen.rl.trainers.ppo import worker_map

    started = time.perf_counter()
    with worker_map(request.workers) as run_map:
        trained = [
            _train_seed(registry, request, config, seed, parent_ref, run_map)
            for seed in request.seeds
        ]
    return trained, time.perf_counter() - started


def opponents_for(registry: Registry, parent_ref: str | None) -> dict[str, Entrant]:
    refs = {
        RANDOM_ROLE: "random@latest",
        GREEDY_ROLE: "greedy@latest",
        BEST_ROLE: parent_ref or "random@latest",
    }
    return {role: registry.resolve(ref).entrant for role, ref in refs.items()}


def selection_score(pairings: Mapping[str, Pairing]) -> float:
    return fmean(pairings[role].win_rate for role in SELECTION_ROLES)


def is_promoted(head_to_head: Pairing) -> bool:
    return head_to_head.ci_low > 0.5


def spread(rates: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(rates),
        "stdev": stdev(rates) if len(rates) > 1 else 0.0,
        "min": min(rates),
        "max": max(rates),
    }


def summarize(
    trained: Sequence[Mapping[str, Any]],
    table: Mapping[str, Mapping[str, Pairing]],
    seconds: float,
) -> dict[str, Any]:
    best = max(trained, key=lambda entry: selection_score(table[entry["ref"]]))
    roles = next(iter(table.values())).keys()
    return {
        "best": dict(best),
        "training_seconds": seconds,
        "summary": {
            role: spread([table[entry["ref"]][role].win_rate for entry in trained])
            for role in roles
        },
        "head_to_head": table[best["ref"]][BEST_ROLE].to_dict(),
        "seeds": [
            {
                **entry,
                "pairings": {role: p.to_dict() for role, p in table[entry["ref"]].items()},
            }
            for entry in trained
        ],
    }


def _evaluate(
    registry: Registry,
    request: StepRequest,
    trained: Sequence[Mapping[str, Any]],
    parent_ref: str | None,
) -> dict[str, dict[str, Pairing]]:
    subjects = [registry.resolve(entry["ref"]).entrant for entry in trained]
    opponents = opponents_for(registry, parent_ref)
    options = {"games": request.games} if request.games else {}
    return run_suite(subjects, opponents, workers=request.workers, **options)


def run_step(registry: Registry, request: StepRequest) -> dict[str, Any]:
    number = step_number(request.folder)
    if any(row["step"] == number for row in registry.experiments()):
        raise RegistryError(f"step {number} already exists")
    parent = _parent_row(registry, request.parent)
    parent_ref = parent["bot_ref"] if parent else None
    config = merged_config(parent, request.overrides, request.budget_change)
    experiment_id = registry.start_experiment(
        step=number,
        parent_ref=parent_ref,
        change=request.change,
        hypothesis=request.hypothesis,
        config=config,
    )
    try:
        trained, seconds = _train_all(registry, request, config, parent_ref)
        table = _evaluate(registry, request, trained, parent_ref)
    except BaseException:
        registry.update_experiment(experiment_id, verdict=FAILED)
        raise
    results = summarize(trained, table, seconds)
    verdict = PROMOTED if is_promoted(table[results["best"]["ref"]][BEST_ROLE]) else REJECTED
    return registry.update_experiment(
        experiment_id,
        bot_ref=results["best"]["ref"],
        train_run_ids=[entry["run_id"] for entry in trained],
        results=json.dumps(results),
        verdict=verdict,
    )


def run_reference(registry: Registry, ref: str, *, workers: int) -> dict[str, Any]:
    best = current_best(registry.experiments())
    subject = registry.resolve(ref).entrant
    opponents = opponents_for(registry, best["bot_ref"] if best else None)
    table = run_suite([subject], opponents, workers=workers)
    pairings = table[subject.ref]
    results = {"pairings": {role: pairing.to_dict() for role, pairing in pairings.items()}}
    return registry.put_reference(
        REFERENCE_ID,
        bot_ref=subject.ref,
        parent_ref=best["bot_ref"] if best else None,
        results=json.dumps(results),
        verdict=REFERENCE,
    )
