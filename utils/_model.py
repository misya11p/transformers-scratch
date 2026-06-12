from importlib import import_module


MODULE_MODELS = "models"


def get_model(config_model):
    cls = getattr(import_module(f"{MODULE_MODELS}"), config_model.name)
    model = cls(**config_model.arch)
    return model
