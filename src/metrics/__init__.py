import torch


def dose_score(prediction, target):
    return torch.abs(prediction - target).sum() / target.sum()


def dvh_score(prediction, target):
    # TODO: what the fuck
    pass
