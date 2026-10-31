import hashlib
import shutil
import tempfile
from pathlib import Path

BOT_FILE = "bot.py"


def code_path(root: Path, sha: str) -> Path:
    return root / "code" / sha / BOT_FILE


def artifact_path(root: Path, sha: str) -> Path:
    return root / "artifacts" / sha


def store_code(root: Path, source: Path) -> str:
    content = source.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    target = code_path(root, sha)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return sha


def store_artifacts(root: Path, source: Path) -> str:
    sha = tree_sha(source)
    target = artifact_path(root, sha)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
            staged = Path(scratch) / sha
            shutil.copytree(source, staged)
            staged.rename(target)
    return sha


def tree_sha(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in source.rglob("*") if path.is_file()):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
