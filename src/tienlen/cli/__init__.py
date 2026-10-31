import typer

app = typer.Typer(help="Tien Len engine, bots, and training.", no_args_is_help=True)


@app.callback()
def main() -> None:
    pass
