import tomllib
from dataclasses import dataclass, field, asdict

from dacite import from_dict


class BaseConfig:
    def asdict(self):
        return asdict(self)


@dataclass
class ModelConfig(BaseConfig):
    arch: str
    tokenizer: str
    max_len: int
    hparams: dict = field(default_factory=dict)

@dataclass
class WandbConfig(BaseConfig):
    log_interval: int
    entity: str
    project: str

@dataclass
class TrainConfig(BaseConfig):
    n_epochs: int
    batch_size: int
    lr: float
    grad_accum_steps: int
    max_grad_norm: float
    wandb: WandbConfig = field(default_factory=WandbConfig)

@dataclass
class Config(BaseConfig):
    model: ModelConfig
    train: TrainConfig


def load_config(fpath_toml) -> Config:
    with open(fpath_toml, "rb") as f:
        config_dict = tomllib.load(f)
    return from_dict(Config, config_dict)
