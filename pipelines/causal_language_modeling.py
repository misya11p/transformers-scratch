import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from .base import BasePipeline


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


class CausalLanguageModelingPipeline(BasePipeline):
    @staticmethod
    def get_tokenizer(fpath_tokenizer):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=fpath_tokenizer)
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
        })
        return tokenizer

    def get_dataset(self):
        tokenizer = self.get_tokenizer(self.config.additional.tokenizer)
        max_len = self.config.model.arch.max_len
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
        get_ds_func = lambda ds: TextDataset(ds, tokenizer, max_len)
        return ds_train, ds_valid, get_ds_func

    def _unpack_batch(self, batch):
        token_ids, seq_ids = batch
        input_ids = token_ids[:, :-1].contiguous().to(self.device)
        labels = token_ids[:, 1:].contiguous().to(self.device)
        seq_ids = seq_ids[:, :-1]
        mask = (seq_ids[:, :, None] != seq_ids[:, None, :]).to(self.device)
        return input_ids, labels, mask

    def _loss_fn(self, pred, labels):
        pred = pred.view(-1, pred.size(-1))
        labels = labels.view(-1)
        loss = F.cross_entropy(pred, labels)
        return loss

    def calc_loss(self, batch):
        input_ids, labels, attention_mask = self._unpack_batch(batch)
        pred = self.model(input_ids, attention_mask)
        loss = self._loss_fn(pred, labels)
        return loss

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = torch.tensor(0., device=self.device)
        n = 0
        for batch in self.valid_loader:
            input_ids, labels, attention_mask = self._unpack_batch(batch)
            with self.context_autocast:
                pred = self.model(input_ids, attention_mask)
                loss = self._loss_fn(pred, labels)
            total_loss += loss
            n += 1
        self.model.train()
        total_loss = self._reduce(total_loss)
        avg_loss = (total_loss / n)
        ppl = torch.exp(avg_loss)
        result = {
            "valid/loss": avg_loss.item(),
            "valid/perplexity": ppl.item(),
        }
        return result
