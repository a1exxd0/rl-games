from string import Template

BOT_TEMPLATE = Template("""\
# /// script
# dependencies = []
# ///
from collections.abc import Sequence
from pathlib import Path

from tienlen.engine import Play, PlayerView, Rules


class $class_name:
    def reset(self, seat: int, rules: Rules, seed: int) -> None:
        pass

    def act(self, view: PlayerView, legal: Sequence[Play]) -> Play:
        return legal[0]


def create_bot(artifact_dir: Path | None) -> $class_name:
    return $class_name()
""")


def render_bot(name: str) -> str:
    words = name.replace("-", "_").replace(".", "_").split("_")
    class_name = "".join(word.capitalize() for word in words) + "Bot"
    return BOT_TEMPLATE.substitute(class_name=class_name)
