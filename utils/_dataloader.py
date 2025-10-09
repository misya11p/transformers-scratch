from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


def get_dataloader(batch_size, tokenizer):
    ds = load_dataset("data/wiki40b_processed/", split="train")
    collater = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collater,
    )
    return train_loader
