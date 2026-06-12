from muon import MuonWithAuxAdam


def get_optimizer(model, hparams_adam=None, hparams_muon=None):
    hparams_adam = hparams_adam or {}
    hparams_muon = hparams_muon or {}
    params_adam = []
    params_muon = []

    for name, parameter in model.named_parameters():
        if (parameter.ndim >= 2) and "transformer_layers." in name:
            params_muon.append(parameter)
        else:
            params_adam.append(parameter)

    optimizer = MuonWithAuxAdam([
        dict(params=params_muon, use_muon=True, **hparams_muon),
        dict(params=params_adam, use_muon=False, **hparams_adam),
    ])

    return optimizer
