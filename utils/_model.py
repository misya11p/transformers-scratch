import torch

from ._tokenizer import get_tokenizer


def get_model(arch: str, hparams: dict) -> torch.nn.Module:
    match arch:
        case "vanilla":
            from models import VanillaTransformer
            model = VanillaTransformer(**hparams)
        case "gpt2":
            from models import GPT2
            model = GPT2(**hparams)
        case _:
            raise ValueError(f"Model {arch} not recognized")
    return model


def load_model(fpath_ckpt, device="cpu"):
    checkpoint = torch.load(fpath_ckpt, map_location="cpu")
    arch = checkpoint["config"]["model"]["arch"]
    hparams = checkpoint["config"]["model"]["hparams"]
    params = checkpoint["model"]
    fpath_tokenizer = checkpoint["config"]["model"]["tokenizer"]
    model = get_model(arch, hparams)
    model.load_state_dict(params)
    model.to(device)
    tokenizer = get_tokenizer(fpath_tokenizer)
    return model, tokenizer
