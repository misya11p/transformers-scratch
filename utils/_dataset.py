from pathlib import Path
import re
import ast

from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._tokenizer import get_tokenizer


DNAME_TEXTS = "texts"
DNAME_TOKENS = "tokens"
FNAME_PARQUET_TRAIN = "train.parquet"
FNAME_PARQUET_VALID = "validation.parquet"


def get_dataloader(
    batch_size,
    dpath_data="data/",
    fpath_tokenizer="tokenizer.json"
):
    dpath_data = Path(dpath_data)
    dpath_tokens = dpath_data / DNAME_TOKENS

    ds_train = load_dataset(str(dpath_tokens), split="train")
    ds_valid = load_dataset(str(dpath_tokens), split="validation")

    tokenizer = get_tokenizer(fpath_tokenizer)
    collater = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collater,
    )
    valid_loader = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collater,
    )
    return train_loader, valid_loader


def format_text(ds):
    sentences = []
    texts = ds["text"]
    for text in tqdm(texts, desc="Formatting text"):
        decoded_string = ast.literal_eval(text).decode("utf-8")
        paragraphs = re.findall(
            r"_START_PARAGRAPH_\n(.*?)(?=\n_START_PARAGRAPH_|\Z)",
            decoded_string,
            re.DOTALL
        )
        for paragraph in paragraphs:
            paragraph = paragraph.replace("_NEWLINE_", "")
            paragraph = paragraph.replace("\n", "")
            paragraph = paragraph.strip()
            if paragraph:
                sentences.append(paragraph)
    ds_new = Dataset.from_dict({"text": sentences})
    return ds_new
