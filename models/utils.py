import torch


def get_model(arch: str, hparams: dict) -> torch.nn.Module:
    match arch:
        case "vanilla_transformer":
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
    model = get_model(arch, hparams)
    model.load_state_dict(params)
    model.to(device)
    return model
