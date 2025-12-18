from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

from ._tokenizer import get_tokenizer


class TextDataset(IterableDataset):
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        for text in self.ds["text"]:
            yield text


def get_dataloader(
    batch_size,
    max_length=1024,
    tokenizer="tokenizer.json",
    world_size=1,
    rank=None,
):
    if isinstance(tokenizer, str):
        tokenizer = get_tokenizer(tokenizer)

    def collate_fn(batch):
        return tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )["input_ids"]

    ds_train = load_dataset(
        "hotchpotch/fineweb-2-edu-japanese",
        "small_tokens_cleaned",
        split="train",
        streaming=True,
    )
    ds_valid = load_dataset(
        "hotchpotch/fineweb-2-edu-japanese",
        "small_tokens_cleaned",
        split="test",
        streaming=True,
    )
    ds_train = ds_train.shuffle(buffer_size=100)

    if world_size >= 2 and rank is not None:
        ds_train = ds_train.shard(num_shards=world_size, index=rank)
        ds_valid = ds_valid.shard(num_shards=world_size, index=rank)

    train_loader = DataLoader(
        TextDataset(ds_train),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )
    valid_loader = DataLoader(
        TextDataset(ds_valid),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    return train_loader, valid_loader
