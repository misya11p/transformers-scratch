from pathlib import Path

from datasets import load_dataset

from utils import format_text, get_tokenizer, train_tokenizer


FPATH_TOKENIZER = "tokenizer.json"
DPATH_DATA = "data/wiki40b/"
DPATH_DATA_PROCESSED = "data/wiki40b_processed/"
fname_parquet = "train.parquet"


def main():
    dpath_data = Path(DPATH_DATA)
    dpath_data_processed = Path(DPATH_DATA_PROCESSED)
    dpath_data.mkdir(parents=True, exist_ok=True)
    dpath_data_processed.mkdir(parents=True, exist_ok=True)
    fpath_tokenizer = Path(FPATH_TOKENIZER)
    fpath_parquet = dpath_data_processed / fname_parquet

    if dpath_data.exists() and any(dpath_data.iterdir()):
        ds = load_dataset(str(dpath_data), split="train")
    else:
        ds = load_dataset("wiki40b", "ja", split="train")
        print("Loaded original dataset.", flush=True)
        ds = format_text(ds)
        ds.to_parquet(dpath_data)
    print("Loaded prepared dataset.", flush=True)

    if not fpath_tokenizer.exists():
        train_tokenizer(
            text=ds["text"],
            fpath_tokenizer=fpath_tokenizer,
            vocab_size=16000,
        )

    if not (
        dpath_data_processed.exists() and any(dpath_data_processed.iterdir())
    ):
        tokenizer = get_tokenizer()
        ds = ds.map(
            lambda x: tokenizer(x["text"]),
            batched=True,
            remove_columns=["text"]
        )
        ds.to_parquet(fpath_parquet)


if __name__ == "__main__":
    main()
