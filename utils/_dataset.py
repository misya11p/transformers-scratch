from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset, DataLoader

from ._tokenizer import get_tokenizer


class TextDataset(IterableDataset):
    def __init__(self, ds, tokenizer, max_length):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        buf_token_ids = []
        buf_seq_ids = []
        seq_id = 0

        for text in self.ds["text"]:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.insert(0, self.tokenizer.bos_token_id)

            buf_token_ids.extend(ids)
            buf_seq_ids.extend([seq_id] * len(ids))
            seq_id += 1

            while len(buf_token_ids) >= self.max_length:
                yield (
                    torch.tensor(buf_token_ids[:self.max_length]),
                    torch.tensor(buf_seq_ids[:self.max_length]),
                )
                buf_token_ids = buf_token_ids[self.max_length:]
                buf_seq_ids = buf_seq_ids[self.max_length:]


def get_dataloader(
    batch_size,
    max_length,
    tokenizer="trained/tokenizer.json",
    world_size=1,
    rank=None,
):
    if isinstance(tokenizer, str):
        tokenizer = get_tokenizer(tokenizer)

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
    )
    ds_train = ds_train.shuffle(buffer_size=10000)

    if (world_size >= 2) and (rank is not None):
        ds_train = ds_train.shard(num_shards=world_size, index=rank)
        ds_valid = ds_valid.shard(num_shards=world_size, index=rank)

    train_loader = DataLoader(
        TextDataset(ds_train, tokenizer, max_length),
        batch_size=batch_size,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        TextDataset(ds_valid, tokenizer, max_length),
        batch_size=batch_size,
        pin_memory=True,
    )

    return train_loader, valid_loader
