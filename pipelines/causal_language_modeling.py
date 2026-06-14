import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from .base import Pipeline


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


class CausalLanguageModelingPipeline(Pipeline):
    @staticmethod
    def get_tokenizer(fpath_tokenizer="trained/tokenizer.json"):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=fpath_tokenizer)
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
        })
        return tokenizer

    def get_dataset(self):
        tokenizer = self.get_tokenizer(self.config.additional["tokenizer"])
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

    @torch.inference_mode()
    def infer(
        self,
        start_text,
        streaming=False,
        max_len=100,
        temperature=1.0,
        top_k=5
    ):
        self.model.eval()
        tokenizer = self.get_tokenizer(self.config.additional["tokenizer"])
        eos_id = tokenizer.eos_token_id
        token_ids = tokenizer.encode(start_text, add_special_tokens=False)
        token_ids.insert(0, eos_id)

        if streaming:
            print(start_text, end="", flush=True)

        for _ in range(max_len):
            input_ids = torch.tensor(
                token_ids[-max_len:],
                device=self.device
            ).unsqueeze(0)

            outputs = self.model(input_ids)
            logits = outputs[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probabilities, 1)]
            next_token = next_token.item()

            token_ids.append(next_token)
            if next_token == eos_id:
                break
            if streaming:
                print(
                    tokenizer.decode([next_token], skip_special_tokens=True),
                    end="",
                    flush=True,
                )

        generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return generated_text
