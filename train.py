import typer

from pipelines import get_pipeline


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    src: str = typer.Argument(
        help=(
            "Path to the config (TOML) file when starting training, or "
            "the path to a checkpoint directory when resuming training."
        ),
    ),
    dpath_ckpt: str = typer.Option(
        None,
        "-c", "--dpath-ckpt",
        help="Path to the directory where checkpoints will be saved.",
    ),
):
    """Train a model."""
    pipeline = get_pipeline(src)
    pipeline.setup_train(dpath_ckpt)
    pipeline.train()


if __name__ == "__main__":
    app()
