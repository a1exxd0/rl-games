import ast
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

ROOTS = ("bots", "scripts", "src", "tests")
INTERFACE_FILE = "interface.py"
DIRECTIVES = ("#!", "# noqa", "# type:")
METADATA_START = "# /// "
METADATA_END = "# ///"
DOCUMENTABLE = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def docstring_lines(path: Path) -> Iterator[int]:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, DOCUMENTABLE) and ast.get_docstring(node) is not None:
            yield node.body[0].lineno


def comment_lines(path: Path) -> Iterator[int]:
    in_metadata = False
    with path.open("rb") as file:
        for token in tokenize.tokenize(file.readline):
            if token.type != tokenize.COMMENT:
                continue
            if in_metadata or token.string.startswith(METADATA_START):
                in_metadata = token.string != METADATA_END
            elif not token.string.startswith(DIRECTIVES):
                yield token.start[0]


def violations(root: Path) -> Iterator[str]:
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for path in paths:
        if path.name == INTERFACE_FILE:
            continue
        yield from (f"{path}:{line}: docstring" for line in docstring_lines(path))
        yield from (f"{path}:{line}: comment" for line in comment_lines(path))


def main() -> int:
    found = [v for root in sys.argv[1:] or ROOTS for v in violations(Path(root))]
    for violation in found:
        print(violation, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
