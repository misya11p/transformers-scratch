import tomllib
from dataclasses import dataclass, asdict

from dacite import from_dict


@dataclass
class ModelConfig:
    arch: str
    hparams: dict
    tokenizer: str = "trained/tokenizer.json"
    max_len: int = 1024

@dataclass
class TrainConfig:
    total_steps: int # Total training steps (n_backward = n_foward / grad_accum_steps)
    batch_size: int # This value will be divided by world_size
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    log_interval: int = 64
    eval_interval: int = 1024
    save_interval: int | None = None
    warmup_ratio: float = 0.1
    muon_params: dict | None = None
    adam_params: dict | None = None
    wandb_project: str = "transformers-scratch"
    wandb_name: str | None = None

@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig

    def asdict(self):
        return asdict(self)


def load_config(name: str) -> Config:
    with open(f"config/{name}.toml", "rb") as f:
        config_dict = tomllib.load(f)
    return from_dict(Config, config_dict)
