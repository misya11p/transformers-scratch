from pathlib import Path
import tomllib
from dataclasses import dataclass, asdict

from dacite import from_dict


ROOT = Path(__file__).parent.parent
DPATH_CONFIG = ROOT / "config"


@dataclass
class TaskConfig:
    name: str

@dataclass
class ModelConfig:
    name: str
    arch: dict

@dataclass
class TrainConfig:
    total_steps: int # Total training steps (n_backward = n_foward / grad_accum_steps)
    batch_size: int # This value will be divided by world_size
    grad_accum_steps: int
    max_grad_norm: float
    warmup_ratio: float
    log_interval: int
    eval_interval: int
    muon: dict
    adam: dict
    save_interval: int | None = None
    wandb_project: str = "deep-learning-scratch"
    wandb_run: str | None = None

@dataclass
class Config:
    task: TaskConfig
    model: ModelConfig
    train: TrainConfig
    additional: dict | None = None

    def asdict(self):
        return asdict(self)


def load_config(name: str) -> Config:
    with open(DPATH_CONFIG / f"{name}.toml", "rb") as f:
        config_dict = tomllib.load(f)
    config = from_dict(Config, config_dict)
    return config
