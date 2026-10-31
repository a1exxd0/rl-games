import hashlib
import importlib.util
import sys
from pathlib import Path

from tienlen.bots.interface import TienLenBot
from tienlen.errors import BotLoadError


def load_bot(path: Path, artifact_dir: Path | None = None) -> TienLenBot:
    module = load_module(Path(path))
    create_bot = getattr(module, "create_bot", None)
    if create_bot is None:
        raise BotLoadError(f"{path}: no create_bot function")
    try:
        bot = create_bot(artifact_dir)
    except Exception as error:
        raise BotLoadError(f"{path}: create_bot failed: {error!r}") from error
    if not isinstance(bot, TienLenBot):
        raise BotLoadError(f"{path}: create_bot must return a TienLenBot")
    return bot


def load_module(path: Path):
    resolved = str(path.resolve())
    name = f"tienlen_bot_{hashlib.sha256(resolved.encode()).hexdigest()[:16]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BotLoadError(f"{path}: not a Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        del sys.modules[name]
        raise BotLoadError(f"{path}: import failed: {error!r}") from error
    return module
