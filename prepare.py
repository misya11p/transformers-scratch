from pathlib import Path

from datasets import load_dataset
import typer

from utils import format_text, get_tokenizer, train_tokenizer


DNAME_TEXTS = "texts"
DNAME_TOKENS = "tokens"
FNAME_PARQUET_TRAIN = "train.parquet"
FNAME_PARQUET_VALID = "validation.parquet"

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    fpath_tokenizer: str = typer.Option(
        "tokenizer.json",
        "-t", "--tokenizer",
        help="File path to save the trained tokenizer"
    ),
    dpath_data: str = typer.Option(
        "data/",
        "-d", "--dir-data",
        help="Directory path to save the processed dataset"
    ),
    vocab_size: int = typer.Option(
        16000,
        "-v", "--vocab-size",
        help="Vocabulary size for the tokenizer"
    ),
):
    """Prepare dataset and tokenizer."""

    fpath_tokenizer = Path(fpath_tokenizer)
    dpath_data = Path(dpath_data)
    dpath_texts = dpath_data / DNAME_TEXTS
    dpath_tokens = dpath_data / DNAME_TOKENS

    skip_download = False
    if dpath_texts.exists():
        files = list(dpath_texts.glob("*.parquet"))
        names = [f.name for f in files]
        if ("train.parquet" in names) and ("validation.parquet" in names):
            skip_download = True
    else:
        dpath_texts.mkdir(parents=True)

    skip_train = False
    if fpath_tokenizer.exists():
        skip_train = True

    skip_tokenize = False
    if dpath_tokens.exists():
        files = list(dpath_tokens.glob("*.parquet"))
        names = [f.name for f in files]
        if ("train.parquet" in names) and ("validation.parquet" in names):
            skip_tokenize = True
    else:
        dpath_tokens.mkdir(parents=True)

    if not skip_download:
        ds_train, ds_valid = load_dataset(
            "wiki40b", "ja", split=["train", "validation"]
        )
        ds_train = format_text(ds_train)
        ds_valid = format_text(ds_valid)
        ds_train.to_parquet(dpath_texts / FNAME_PARQUET_TRAIN)
        ds_valid.to_parquet(dpath_texts / FNAME_PARQUET_VALID)
        print("Saved formatted texts.", flush=True)
    else:
        dss = load_dataset(str(dpath_texts))
        ds_train = dss["train"]
        ds_valid = dss["validation"]
        print("Loaded formatted texts.", flush=True)

    if not skip_train:
        train_tokenizer(
            text=ds_train["text"],
            fpath_tokenizer=fpath_tokenizer,
            vocab_size=vocab_size,
        )

    if not (skip_train and skip_tokenize):
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
        ds_train.to_parquet(dpath_tokens / FNAME_PARQUET_TRAIN)
        ds_valid.to_parquet(dpath_tokens / FNAME_PARQUET_VALID)
        print("Saved tokenized dataset.", flush=True)


if __name__ == "__main__":
    app()
