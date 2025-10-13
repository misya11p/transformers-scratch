from pathlib import Path
import os
import shutil
from datetime import datetime, timedelta, timezone
from collections import OrderedDict
import contextlib

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from schedulefree import RAdamScheduleFree
import typer

from utils import load_config, get_tokenizer, get_dataloader


JST = timezone(timedelta(hours=9))
FNAME_LATEST = "latest.pth"

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    fpath_config: str = typer.Option(
        "config.toml",
        "-c", "--config",
        help="File path to the config file (.toml)",
    ),
    dpath_data: str = typer.Option(
        "data/",
        "-d", "--data",
        help="Directory path to the dataset",
    ),
    dpath_ckpt: str = typer.Option(
        "checkpoints/",
        "-k", "--checkpoints",
        help="Directory path to save checkpoints",
    ),
):
    """Train a language model."""

    config = load_config(fpath_config)
    dpath_data = Path(dpath_data)
    dpath_ckpt = Path(dpath_ckpt)
    trainer = Trainer(config, dpath_data, dpath_ckpt)
    trainer.train()


class Trainer:
    def __init__(self, config, dpath_data, dpath_ckpt):
        self.start_time = datetime.now(JST)
        self.tokenizer = get_tokenizer(config.model.tokenizer)

        config.model.hparams["max_len"] = config.model.max_len
        config.model.hparams["vocab_size"] = self.tokenizer.vocab_size
        self.config = config.asdict()

        arch = config.model.arch
        match arch:
            case "vanilla":
                from models import VanillaTransformer
                self.model = VanillaTransformer(**config.model.hparams)
            case _:
                raise ValueError(f"Model {arch} not recognized")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = RAdamScheduleFree(
            self.model.parameters(),
            lr=config.train.lr,
            betas=config.train.betas,
        )
        self.scaler = torch.amp.GradScaler()

        self.max_len = config.model.max_len
        self.n_epochs = config.train.n_epochs
        self.grad_accum_steps = config.train.grad_accum_steps
        self.max_grad_norm = config.train.max_grad_norm
        self.epoch = 0

        self._setup_device()
        self.model = torch.compile(self.model)

        self.train_loader, self.valid_loader = get_dataloader(
            batch_size=config.train.batch_size,
            dpath_data=dpath_data,
            tokenizer=self.tokenizer,
            world_size=self.world_size,
            rank=self.global_rank,
        )
        self.n_iter_per_epoch = len(self.train_loader)
        self._setup_checkpoint(dpath_ckpt)

        if self.is_master:
            print(f"Model: {config.model.arch}")
            n_params = sum(
                p.numel()
                for p in self.model.parameters()
                if p.requires_grad
            )
            print(f"Number of parameters: {n_params:,}")
            print("Number of devices:", self.world_size)
            print(f"Checkpoints will be saved to {self.dpath_ckpt}")

            if config.train.wandb.log_interval > 0:
                import wandb
                name = (
                    config.train.wandb.name
                    or f"[{arch}] {self.start_time.strftime('%m/%d %H:%M')}"
                )
                self.wandb_run = wandb.init(
                    project=config.train.wandb.project,
                    name=name,
                    config=self.config,
                )

    def _is_dist(self):
        return (
            dist.is_available()
            and torch.cuda.is_available()
            and os.environ.get("RANK") is not None
            and int(os.environ["WORLD_SIZE"]) > 1
        )

    def _setup_device(self):
        if self._is_dist():
            torch.accelerator.set_device_index(rank)
            acc = torch.accelerator.current_accelerator()
            backend = torch.distributed.get_default_backend_for_device(acc)
            dist.init_process_group(backend)
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_global_rank()
            rank = dist.get_rank()
            self.device = torch.device(rank)
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.world_size = 1
            self.global_rank = 1
            self.device = torch.accelerator.current_accelerator()
            self.model = self.model.to(self.device)

        self.is_master = self.global_rank == 0

    def _setup_checkpoint(self, dpath_ckpt_root):
        dpath_ckpt_root.mkdir(parents=True, exist_ok=True)
        datestr = self.start_time.strftime("%Y%m%d_%H%M%S")
        dpath_ckpt = dpath_ckpt_root / datestr
        self.dpath_ckpt = Path(dpath_ckpt)
        self.dpath_ckpt.mkdir(parents=True, exist_ok=True)
        self.fpath_latest = dpath_ckpt_root / FNAME_LATEST

    def train(self):
        if self.is_master:
            print("Training started.", flush=True)
        model = self.model # alias
        model.train()

        for epoch in range(self.n_epochs):
            self.epoch = epoch + 1
            for i, batch in enumerate(self.train_loader, 1):
                input_ids, labels = self._unpack_batch(batch)

                is_update_step = (
                    i % self.grad_accum_steps == 0
                    or i == self.n_iter_per_epoch
                )
                if is_update_step:
                    context_nosync = contextlib.nullcontext()
                else:
                    context_nosync = model.no_sync()

                context_autocast = torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                )

                with context_nosync, context_autocast:
                    pred = model(input_ids)
                    loss = self._loss_fn(pred, labels)
                loss = self.scaler(loss / self.grad_accum_steps)
                loss.backward()

                if is_update_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            self._save_checkpoint()

    def _unpack_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        input_ids = input_ids[:, :self.max_len - 1].contiguous()
        labels = labels[:, 1:self.max_len].contiguous()
        return input_ids, labels

    def _loss_fn(self, pred, labels):
        pred = pred.view(-1, pred.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(pred, labels)
        return loss

    def _save_checkpoint(self):
        state_dict = self.model.state_dict()
        correct_state_dict = OrderedDict()
        for key, value in state_dict.items():
            key = key.replace("_orig_mod.", "")
            key = key.replace("module.", "")
            correct_state_dict[key] = value
        state_dict = {
            "model": correct_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        fpath_state_dict = self.dpath_ckpt / f"epoch{self.epoch}.pth"
        torch.save(state_dict, fpath_state_dict)
        shutil.copy(fpath_state_dict, self.fpath_latest)

if __name__ == "__main__":
    main()
