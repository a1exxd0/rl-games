import typer
from typer.core import TyperGroup

from tienlen.cli import arena, bot, db, dev, lab, match, train, web
from tienlen.errors import TienLenError


class ErrorGroup(TyperGroup):
    def invoke(self, ctx: typer.Context) -> object:  # type: ignore
        try:
            return super().invoke(ctx)
        except TienLenError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error


app = typer.Typer(
    cls=ErrorGroup,
    help="Tien Len engine, bots, and training.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.callback()
def main() -> None:
    pass


for name, group in (
    ("bot", bot),
    ("match", match),
    ("arena", arena),
    ("train", train),
    ("lab", lab),
    ("db", db),
    ("web", web),
    ("dev", dev),
):
    app.add_typer(group.app, name=name)
