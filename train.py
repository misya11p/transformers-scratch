import typer

from pipelines import get_pipeline


FNAME_STATE = "state.pth"
FNAME_MODEL = "model.safetensors"

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    src: str = typer.Argument(
        help="",
    ),
    dpath_ckpt: str = typer.Option(
        None,
        "-c", "--dpath-ckpt",
        help="",
    ),
):
    """Train a model."""
    pipeline = get_pipeline(src)
    pipeline.setup_train(dpath_ckpt)
    pipeline.train()


if __name__ == "__main__":
    app()
