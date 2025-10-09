from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import tensorboard
from tqdm import tqdm

from utils import get_tokenizer, get_dataloader


JST = timezone(timedelta(hours=9))
FPATH_LATEST = "checkpoints/latest.pth"
DNAME_LOGS = "logs"
FPATH_CONFIG = "config.json"
with open(FPATH_CONFIG, "r") as f:
    config = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    trainer = Trainer(config)
    trainer.train()


class Trainer:
    def __init__(self, config):
        model = config["model"]
        model_config = config["model_config"]
        train_config = config["train_config"]

        self.tokenizer = get_tokenizer()
        model_config["vocab_size"] = self.tokenizer.vocab_size

        if model == "vanilla":
            from models import VanillaTransformer
            self.model = VanillaTransformer(**model_config)
            print("Model: Vanilla Transformer", flush=True)
        else:
            raise ValueError(f"Model {model} not recognized")

        self.max_len = model_config["max_len"]
        self.n_epochs = train_config["n_epochs"]
        self.learning_rate = train_config["learning_rate"]
        self.batch_size = train_config["batch_size"]

        self.dataloader = get_dataloader(
            batch_size=self.batch_size,
            tokenizer=self.tokenizer
        )

        now = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        dpath_ckpt = f"checkpoints/{now}/"
        self.dpath_ckpt = Path(dpath_ckpt)
        self.dpath_ckpt.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=len(self.dataloader) * self.n_epochs
        )
        dpath_logs = self.dpath_ckpt / DNAME_LOGS
        dpath_logs.mkdir(parents=True, exist_ok=True)
        self.writer = tensorboard.SummaryWriter(log_dir=str(dpath_logs))
        print(f"Checkpoints will be saved to {self.dpath_ckpt}", flush=True)

        subprocess.Popen(
            [
                "tensorboard",
                "--logdir", str(dpath_logs),
                "--host", "0.0.0.0",
                "--port", "6006"
            ],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        print("TensorBoard started.", flush=True)

        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        print("Training started.", flush=True)
        self.model.to(self.device)
        self.model.train()
        n_iter = 0
        for epoch in range(self.n_epochs):
            self.epoch = epoch + 1
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {self.epoch}/{self.n_epochs}",
                leave=False
            )
            for batch in pbar:
                input_ids, labels = self._unpack_batch(batch)
                pred = self.model(input_ids)
                loss = self._calc_loss(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.writer.add_scalar("loss", loss.item(), n_iter)
                n_iter += 1
            self._save_checkpoint()

    def _unpack_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        input_ids = input_ids[:, :self.max_len].contiguous()
        labels = labels[:, :self.max_len].contiguous()
        return input_ids, labels

    def _calc_loss(self, pred, labels):
        pred = pred.view(-1, pred.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(pred, labels)
        return loss

    def _save_checkpoint(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(state_dict, self.dpath_ckpt / f"epoch{self.epoch}.pth")
        torch.save(state_dict, FPATH_LATEST)

if __name__ == "__main__":
    main()