# /// script
# dependencies = ["numpy"]
# ///
from pathlib import Path

from tienlen.rl.policy import WEIGHTS_FILE, Policy, PolicyBot


def create_bot(artifact_dir: Path | None) -> PolicyBot:
    if artifact_dir is None:
        raise ValueError(f"neural bot needs an artifact_dir holding {WEIGHTS_FILE}")
    return PolicyBot(Policy.load(Path(artifact_dir) / WEIGHTS_FILE))
