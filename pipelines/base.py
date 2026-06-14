import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
from importlib import import_module
from collections import OrderedDict
import contextlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.accelerator import current_accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from muon import MuonWithAuxAdam
from safetensors.torch import save_file
import wandb
from tqdm import tqdm


ROOT = Path(__file__).parent.parent
DPATH_CHECKPOINTS = ROOT / "checkpoints"
MODULE_MODELS = "models"
FNAME_STATE = "state.pth"
FNAME_MODEL = "model.safetensors"
CPU = torch.device("cpu")


class Pipeline(ABC):
    def __init__(self, config, state_dict_model=None):
        self.config = config
        self.device = current_accelerator(check_available=True) or CPU
        self.model = self.get_model()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if state_dict_model is not None:
            self.model.load_state_dict(state_dict_model)
        if not self._is_dist(): # Prevent the same model from being placed on multiple devices
            self.model.to(self.device)

    def get_model(self):
        cls = getattr(import_module(f"{MODULE_MODELS}"), self.config.model.name)
        model = cls(**self.config.model.arch)
        return model

    def setup_train(self, dpath_ckpt=None):
        self.start_time = datetime.now(tz=ZoneInfo("Asia/Tokyo"))
        config_train = self.config.train # alias

        self.total_steps = config_train.total_steps
        self.grad_accum_steps = config_train.grad_accum_steps
        scheduler_steps = config_train.total_steps // config_train.grad_accum_steps
        warmup_steps = int(config_train.warmup_ratio * scheduler_steps)
        self.max_grad_norm = config_train.max_grad_norm

        self.optimizer = self._get_optimizer()
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_steps,
        )
        self.scaler = torch.amp.GradScaler()

        self.now_steps = 0
        self._setup_checkpoint(dpath_ckpt)
        self.is_dist = self._is_dist()
        self._setup_device()
        self.is_master = self.global_rank == 0
        self.context_autocast = torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

        self.model = torch.compile(self.model)

        self.train_loader, self.valid_loader = self.get_dataloader()
        self.log_interval = config_train.log_interval
        self.eval_interval = config_train.eval_interval
        self.save_interval = config_train.save_interval

        if self.is_master:
            print(f"Model: {self.config.model.name}")
            print(f"Number of parameters: {self.n_params:,}")
            print(f"Number of devices: {self.world_size}")
            if self.resume:
                print(f"Resumed from checkpoint at {self.dpath_ckpt}")
            else:
                print(f"Checkpoints will be saved to {self.dpath_ckpt}")

            if config_train.wandb_run:
                name = config_train.wandb_run
            else:
                name = (
                    f"[{self.config.model.name} {self.n_params // 1_000_000}M] "
                    f"{self.start_time.strftime('%Y-%m-%d %H:%M')}"
                )

            self.wandb_run = wandb.init(
                project=config_train.wandb_project,
                name=name,
                config=self.config,
            )

    def _setup_checkpoint(self, dpath_ckpt):
        if dpath_ckpt is None:
            datestr = self.start_time.strftime("%Y%m%d_%H%M%S")
            dpath_ckpt = DPATH_CHECKPOINTS / datestr
            dpath_ckpt.mkdir(parents=True, exist_ok=True)
            self.resume = False
        else:
            dpath_ckpt = Path(dpath_ckpt)
            assert dpath_ckpt.exists(),\
                f"The checkpoint directory {dpath_ckpt} does not exist."

            latest_ckpt = dpath_ckpt / FNAME_STATE
            state_dict = torch.load(latest_ckpt, map_location=CPU)
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])
            self.scaler.load_state_dict(state_dict["scaler"])
            self.now_steps = state_dict["now_steps"]
            self.resume = True
        self.dpath_ckpt = dpath_ckpt

    @staticmethod
    def _is_dist():
        world_size = os.environ.get("WORLD_SIZE", 0)
        return (
            dist.is_available()
            and torch.cuda.is_available()
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
            self.model = self.model.to(self.device)

    def _get_optimizer(self):
        params_adam = []
        params_muon = []

        for name, parameter in self.model.named_parameters():
            if (parameter.ndim >= 2) and "transformer_layers." in name:
                params_muon.append(parameter)
            else:
                params_adam.append(parameter)

        optimizer = MuonWithAuxAdam([
            dict(params=params_muon, use_muon=True, **self.config_train.muon),
            dict(params=params_adam, use_muon=False, **self.config_train.adam),
        ])

        return optimizer

    def _get_dataloader(self):
        ds_train, ds_valid, get_ds_func = self.get_dataset()
        ds_train = ds_train.shuffle(buffer_size=10000)

        if (self.world_size >= 2) and (self.global_rank is not None):
            ds_train = ds_train.shard(
                num_shards=self.world_size,
                index=self.global_rank,
            )
            ds_valid = ds_valid.shard(
                num_shards=self.world_size,
                index=self.global_rank,
            )

        train_loader = DataLoader(
            get_ds_func(ds_train),
            batch_size=self.config.train.batch_size,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            get_ds_func(ds_valid),
            batch_size=self.config.train.batch_size,
            pin_memory=True,
        )
        return train_loader, valid_loader

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
                    is_saving_step = self.save_interval is not None
                else:
                    is_updating_step = self.now_steps % self.grad_accum_steps == 0
                    is_logging_step = self.now_steps % self.log_interval == 0
                    is_evaluating_step = self.now_steps % self.eval_interval == 0
                    is_saving_step = (
                        self.save_interval
                        and self.now_steps % self.save_interval == 0
                    )

                if (not is_updating_step) and self.is_dist:
                    context_nosync = self.model.no_sync()
                else:
                    context_nosync = contextlib.nullcontext()

                with context_nosync, self.context_autocast:
                    loss = self.calc_loss(batch)
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
                        self.wandb_run.log(
                            { "train/loss": loss },
                            step=self.now_steps,
                        )
                        self._save_checkpoint()

                if is_evaluating_step:
                    result_eval = self.evaluate()
                    if self.is_master:
                        self.wandb_run.log(result_eval, step=self.now_steps)
                        self._save_checkpoint()

                if is_saving_step:
                    if self.is_master:
                        self._save_checkpoint(snapshot=True)

                if is_last:
                    is_running = False
                    break

        if self.is_master:
            print("Training finished.", flush=True)
            self.wandb_run.finish()

    def _reduce(self, tensor, avg=True):
        if self.is_dist:
            dist.reduce(tensor, dst=0)
            if avg:
                tensor = tensor / self.world_size
        return tensor

    def _save_checkpoint(self, snapshot=False):
        model_state_dict = self.model.state_dict()
        correct_model_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            key = key.replace("_orig_mod.", "")
            key = key.replace("module.", "")
            correct_model_state_dict[key] = value

        save_file(correct_model_state_dict, self.dpath_ckpt / FNAME_MODEL)

        state_dict = {
            "model": correct_model_state_dict,
            "now_steps": self.now_steps,
            "config": self.config.asdict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(state_dict, self.dpath_ckpt / FNAME_STATE)

        if snapshot:
            fname_snapshot = f"{self.now_steps:0{len(str(self.total_steps))}d}.pth"
            fpath_snapshot = self.dpath_ckpt / fname_snapshot
            torch.save(state_dict, fpath_snapshot)

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def calc_loss(self, batch):
        pass

    @abstractmethod
    def evaluate(self):
        pass
