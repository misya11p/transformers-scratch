import torch
from safetensors.torch import load_file

from ._config import load_config
from ._tokenizer import get_tokenizer


def get_model(fname_config) -> torch.nn.Module:
    config = load_config(fname_config)

    tokenizer = get_tokenizer(config.model.tokenizer)
    arch = config.model.arch
    hparams = config.model.hparams
    hparams["max_len"] = config.model.max_len
    hparams["vocab_size"] = tokenizer.vocab_size

    match arch:
        case "vanilla":
            from models import VanillaTransformer
            model = VanillaTransformer(**hparams)
        case "gpt2":
            from models import GPT2
            model = GPT2(**hparams)
        case _:
            raise ValueError(f"Model {arch} not recognized")

    return model, tokenizer, config


def load_model(fpath_ckpt, fname_config, device="cpu"):
    model, tokenizer, _ = get_model(fname_config)
    model.load_state_dict(load_file(fpath_ckpt, device="cpu"))
    model.to(device)
    return model, tokenizer