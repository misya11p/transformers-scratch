from pathlib import Path

from datasets import load_dataset
import typer

from utils import get_tokenizer, train_tokenizer
from utils import (
    FNAME_PARQUET_TRAIN,
    FNAME_PARQUET_VALID,
)


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    fpath_tokenizer: str = typer.Option(
        "tokenizer.json",
        "-t", "--tokenizer",
        help="File path to save the trained tokenizer"
    ),
    vocab_size: int = typer.Option(
        16000,
        "--vocab-size",
        help="Vocabulary size for the tokenizer"
    ),
    dpath_data: str = typer.Option(
        "data/",
        "--dir-data",
        help="Directory path to save the processed dataset"
    ),
    dpath_cache: str = typer.Option(
        "cache/",
        "--dir-cache",
        help="Directory path to cache the downloaded dataset"
    ),
    max_rows: int = typer.Option(
        1_000_000,
        "--max-rows",
        help="Maximum number of rows to use for training the tokenizer"
    ),
):
    """Prepare dataset and tokenizer."""

    fpath_tokenizer = Path(fpath_tokenizer)
    dpath_data = Path(dpath_data)
    dpath_data.mkdir(parents=True, exist_ok=True)

    dss = load_dataset("fujiki/wiki40b_ja", cache_dir=dpath_cache)
    ds_train = dss["train"]
    ds_valid = dss["test"]

    if not fpath_tokenizer.exists():
        train_tokenizer(
            text=ds_train["text"],
            fpath_tokenizer=fpath_tokenizer,
            vocab_size=vocab_size,
            max_rows=max_rows,
        )

    tokenizer = get_tokenizer()
    ds_train = ds_train.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=["text"]
    )
    ds_valid = ds_valid.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=["text"]
    )
    ds_train.to_parquet(dpath_data / FNAME_PARQUET_TRAIN)
    ds_valid.to_parquet(dpath_data / FNAME_PARQUET_VALID)
    print("Saved tokenized dataset.", flush=True)


if __name__ == "__main__":
    app()
