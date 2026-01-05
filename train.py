import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import OrderedDict
import contextlib

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
import wandb
import typer
from tqdm import tqdm

from utils import load_config, get_tokenizer, get_dataloader, get_optimizer
from models import get_model


FANME_LATEST = "latest.pth"
JST = timezone(timedelta(hours=9))
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    fname_config: str = typer.Option(
        None,
        "-c", "--config",
        help="File name of the config file (.toml) in ./config",
    ),
    dpath_ckpt: str = typer.Option(
        None,
        "-k", "--checkpoints",
        help="Directory path to save checkpoints",
    ),
):
    """Train a language model."""
    config = load_config(fname_config)
    trainer = Trainer(config, dpath_ckpt)
    trainer.train()


class Trainer:
    def __init__(self, config, dpath_ckpt):
        self.start_time = datetime.now(JST)
        self.tokenizer = get_tokenizer(config.model.tokenizer)

        config.model.hparams["max_len"] = config.model.max_len
        config.model.hparams["vocab_size"] = self.tokenizer.vocab_size
        self.config = config.asdict()

        self.model = get_model(config.model.arch, config.model.hparams)
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(
            self.model,
            hparams_muon=config.train.muon_params,
            hparams_adam=config.train.adam_params,
        )

        self.total_steps = config.train.total_steps
        self.grad_accum_steps = config.train.grad_accum_steps
        scheduler_steps = config.train.total_steps // self.grad_accum_steps
        warmup_steps = int(config.train.warmup_ratio * scheduler_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_steps,
        )
        self.scaler = torch.amp.GradScaler()

        self.now_tokens = 0
        self.now_steps = 0
        self._setup_checkpoint(dpath_ckpt)

        self.max_len = config.model.max_len
        self.max_grad_norm = config.train.max_grad_norm
        self.is_dist = self._is_dist()
        self._setup_device()
        self.is_master = self.global_rank == 0
        self.context_autocast = torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

        self.model = torch.compile(self.model)
        self.train_loader, self.valid_loader = get_dataloader(
            batch_size=config.train.batch_size // self.world_size,
            max_length=self.max_len + 1,
            tokenizer=self.tokenizer,
            world_size=self.world_size,
            rank=self.global_rank,
        )
        self.log_interval = config.train.log_interval
        self.eval_interval = config.train.eval_interval
        self.save_interval = config.train.save_interval

        if self.is_master:
            print(f"Model: {config.model.arch}")
            print(f"Number of parameters: {self.n_params:,}")
            print(f"Number of devices: {self.world_size}")
            if self.resume:
                print(f"Resumed from checkpoint at {self.dpath_ckpt}")
            else:
                print(f"Checkpoints will be saved to {self.dpath_ckpt}")

            if config.train.wandb_name:
                name = config.train.wandb_name
            else:
                name = (
                    f"[{config.model.arch} {self.n_params // 1_000_000}M] "
                    f"{self.start_time.strftime('%Y-%m-%d %H:%M')}"
                )

            self.wandb_run = wandb.init(
                project=config.train.wandb_project,
                name=name,
                config=self.config,
            )
            self.wandb_run.define_metric(
                "train/perplexity",
                step_metric="total_tokens"
            )
            self.wandb_run.define_metric(
                "valid/perplexity",
                step_metric="total_tokens"
            )

    @staticmethod
    def _is_dist():
        world_size = os.environ.get("WORLD_SIZE")
        return (
            dist.is_available()
            and torch.cuda.is_available()
            and world_size is not None
            and int(world_size) >= 2
        )

    def _setup_device(self):
        if self.is_dist:
            rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            torch.accelerator.set_device_index(rank)
            acc = torch.accelerator.current_accelerator()
            backend = torch.distributed.get_default_backend_for_device(acc)
            dist.init_process_group(backend)
            self.world_size = dist.get_world_size()
            self.device = torch.device(rank)
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[rank])
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.accelerator.current_accelerator(check_available=True)
            self.model = self.model.to(self.device)

    def _setup_checkpoint(self, dpath_ckpt):
        if dpath_ckpt is None:
            datestr = self.start_time.strftime("%Y%m%d_%H%M%S")
            dpath_ckpt = Path("checkpoints") / datestr
            dpath_ckpt.mkdir(parents=True, exist_ok=True)
            self.resume = False
        else:
            dpath_ckpt = Path(dpath_ckpt)
            assert dpath_ckpt.exists(), f"The checkpoint directory {dpath_ckpt} does not exist."

            latest_ckpt = dpath_ckpt / FANME_LATEST
            state_dict = torch.load(latest_ckpt, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])
            self.scaler.load_state_dict(state_dict["scaler"])
            self.now_steps = state_dict["now_steps"]
            self.now_tokens = state_dict["now_tokens"]
            self.resume = True
        self.dpath_ckpt = dpath_ckpt
        self.fpath_latest = self.dpath_ckpt / FANME_LATEST

    def train(self):
        if self.is_master:
            print("Training started.", flush=True)
        self.model.train()

        is_running = True
        pbar = tqdm(total=self.total_steps, disable=not self.is_master)
        pbar.n = self.now_steps
        pbar.refresh()
        while is_running:
            for batch in self.train_loader:
                pbar.update()
                self.now_steps += 1
                is_last = self.now_steps >= self.total_steps
                if is_last:
                    is_updating_step = True
                    is_logging_step = True
                    is_evaluating_step = True
                    is_saving_step = True
                else:
                    is_updating_step = self.now_steps % self.grad_accum_steps == 0
                    is_logging_step = (
                        (self.now_steps >= self.eval_interval)
                        and (self.now_steps % self.log_interval == 0)
                    )
                    is_evaluating_step = self.now_steps % self.eval_interval == 0
                    is_saving_step = self.now_steps % self.save_interval == 0

                input_ids, labels, attention_mask = self._unpack_batch(batch)
                self.now_tokens += input_ids.numel() * self.world_size

                if (not is_updating_step) and self.is_dist:
                    context_nosync = self.model.no_sync()
                else:
                    context_nosync = contextlib.nullcontext()

                with context_nosync, self.context_autocast:
                    pred = self.model(input_ids, attention_mask)
                    loss = self._loss_fn(pred, labels)
                loss_scaled = self.scaler.scale(loss / self.grad_accum_steps)
                loss_scaled.backward()

                if is_updating_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                if is_logging_step:
                    loss = self._reduce(loss.detach())
                    if self.is_master:
                        ppl = torch.exp(loss).item()
                        self.wandb_run.log(
                            {
                                "train/perplexity": ppl,
                                "total_tokens": self.now_tokens,
                            },
                            step=self.now_steps,
                        )
                        self._save_checkpoint(latest_only=True)

                if is_evaluating_step:
                    ppl = self._evaluate()
                    if self.is_master:
                        self.wandb_run.log(
                            {
                                "valid/perplexity": ppl,
                                "total_tokens": self.now_tokens,
                            },
                            step=self.now_steps,
                        )
                        print(
                            f"[{datetime.now(JST).strftime('%m/%d %H:%M:%S')}] "
                            f"Step {self.now_steps}/{self.total_steps} - "
                            f"Validation Perplexity: {ppl:.2f}",
                            flush=True,
                        )
                        self._save_checkpoint(latest_only=True)

                if is_saving_step:
                    if self.is_master:
                        self._save_checkpoint(latest_only=False)

                if is_last:
                    is_running = False
                    break

        if self.is_master:
            print("Training finished.", flush=True)
            self.wandb_run.finish()

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
        loss = self.criterion(pred, labels)
        return loss

    def _reduce(self, tensor, avg=True):
        if self.is_dist:
            dist.reduce(tensor, dst=0)
            if avg:
                tensor = tensor / self.world_size
        return tensor

    @torch.no_grad()
    def _evaluate(self):
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
        total_loss = self._reduce(total_loss)
        avg_loss = total_loss / n
        ppl = torch.exp(avg_loss).item()
        self.model.train()
        return ppl

    def _save_checkpoint(self, latest_only=True):
        state_dict = self.model.state_dict()
        correct_state_dict = OrderedDict()
        for key, value in state_dict.items():
            key = key.replace("_orig_mod.", "")
            key = key.replace("module.", "")
            correct_state_dict[key] = value
        state_dict = {
            "model": correct_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "now_steps": self.now_steps,
            "now_tokens": self.now_tokens,
            "config": self.config,
        }
        torch.save(state_dict, self.fpath_latest)
        if not latest_only:
            fname_state_dict = f"{self.now_steps:0{len(str(self.total_steps))}d}.pth"
            fpath_state_dict = self.dpath_ckpt / fname_state_dict
            torch.save(state_dict, fpath_state_dict)


if __name__ == "__main__":
    app()
