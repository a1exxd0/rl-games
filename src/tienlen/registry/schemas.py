import pyarrow as pa

NAMESPACE = "tienlen"
TIMESTAMP = pa.timestamp("us", tz="UTC")

BOTS = pa.schema(
    [
        ("bot", pa.string()),
        ("kind", pa.string()),
        ("description", pa.string()),
        ("created_at", TIMESTAMP),
    ]
)

BOT_VERSIONS = pa.schema(
    [
        ("bot", pa.string()),
        ("version", pa.int64()),
        ("kind", pa.string()),
        ("code_sha", pa.string()),
        ("artifact_sha", pa.string()),
        ("parent_version", pa.string()),
        ("train_run_id", pa.string()),
        ("rules_id", pa.string()),
        ("engine_version", pa.string()),
        ("browser_compatible", pa.bool_()),
        ("status", pa.string()),
        ("created_at", TIMESTAMP),
    ]
)

MATCHES = pa.schema(
    [
        ("match_id", pa.string()),
        ("arena_id", pa.string()),
        ("seed", pa.int64()),
        ("rules_id", pa.string()),
        ("engine_version", pa.string()),
        ("seats", pa.list_(pa.string())),
        ("placements", pa.list_(pa.int32())),
        ("forfeits", pa.list_(pa.int32())),
        ("turns", pa.int32()),
        ("replay_uri", pa.string()),
        ("played_at", TIMESTAMP),
    ]
)

RATINGS = pa.schema(
    [
        ("bot", pa.string()),
        ("version", pa.int64()),
        ("rules_id", pa.string()),
        ("n_players", pa.int32()),
        ("mu", pa.float64()),
        ("sigma", pa.float64()),
        ("games", pa.int64()),
        ("computed_at", TIMESTAMP),
    ]
)

TRAIN_RUNS = pa.schema(
    [
        ("run_id", pa.string()),
        ("bot", pa.string()),
        ("config", pa.string()),
        ("status", pa.string()),
        ("metrics_uri", pa.string()),
        ("started_at", TIMESTAMP),
        ("finished_at", TIMESTAMP),
    ]
)

EXPERIMENTS = pa.schema(
    [
        ("experiment_id", pa.string()),
        ("step", pa.int32()),
        ("bot_ref", pa.string()),
        ("parent_ref", pa.string()),
        ("change", pa.string()),
        ("hypothesis", pa.string()),
        ("config", pa.string()),
        ("train_run_ids", pa.list_(pa.string())),
        ("results", pa.string()),
        ("verdict", pa.string()),
        ("notes", pa.string()),
        ("created_at", TIMESTAMP),
    ]
)

TABLES = {
    "bots": BOTS,
    "bot_versions": BOT_VERSIONS,
    "matches": MATCHES,
    "ratings": RATINGS,
    "train_runs": TRAIN_RUNS,
    "experiments": EXPERIMENTS,
}
