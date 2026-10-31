import ast
import re
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

SCRIPT_BLOCK = re.compile(
    r"(?m)^# /// script$\s(?P<content>(^#(| .*)$\s)+)^# ///$",
)
DISTRIBUTION_NAME = re.compile(r"[A-Za-z0-9._-]+")
BROWSER_PACKAGES = frozenset({"numpy", "tienlen"})


def declared_dependencies(source: str) -> list[str]:
    block = SCRIPT_BLOCK.search(source)
    if block is None:
        return []
    lines = (line.removeprefix("#").removeprefix(" ") for line in block["content"].splitlines())
    return tomllib.loads("\n".join(lines)).get("dependencies", [])


def browser_compatible(path: Path) -> bool:
    source = path.read_text()
    declared = {_distribution(spec) for spec in declared_dependencies(source)}
    imported = set(_imported_modules(source)) - set(sys.stdlib_module_names)
    return (declared | imported) <= BROWSER_PACKAGES


def _distribution(spec: str) -> str:
    match = DISTRIBUTION_NAME.match(spec)
    if match is None:
        raise ValueError(f"invalid dependency: {spec}")
    return match[0].lower().replace("_", "-")


def _imported_modules(source: str) -> Iterator[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            yield from (alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.partition(".")[0]
