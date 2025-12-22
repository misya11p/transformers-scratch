import tomllib
from dataclasses import dataclass, asdict

from dacite import from_dict


@dataclass
class ModelConfig:
    arch: str
    hparams: dict
    tokenizer: str = "tokenizer.json"
    max_len: int = 1024

@dataclass
class TrainConfig:
    total_steps: int # Total training steps (n_backward = n_foward / grad_accum_steps)
    batch_size: int # This value will be divided by world_size
    lr: float | None = None
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1000
    wandb_name: str | None = None
    wandb_project: str = "transformers-scratch"

@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig

    def asdict(self):
        return asdict(self)


def load_config(fpath_toml="config.toml") -> Config:
    with open(fpath_toml, "rb") as f:
        config_dict = tomllib.load(f)
    return from_dict(Config, config_dict)
