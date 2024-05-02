import torch
from pathlib import Path
from typing import Optional, TypeVar

import matplotlib.pyplot as plt
from torch._C import _EnablePythonDispatcher

from src.data import DataLoader, get_paths
from src.models import ConvNet
from src.training import train

T = TypeVar("T")


def unwrap(var: Optional[T]) -> T:
    if var is None:
        raise Exception("Could not unwrap")

    return var


def run():
    primary_directory = Path().resolve()
    provided_data_dir = primary_directory / "data"
    training_data_dir = provided_data_dir / "train-pats"
    training_plan_paths = get_paths(training_data_dir)
    data_loader_train = DataLoader(training_plan_paths)

    data_loader_train.set_mode("training_model")
    model = ConvNet()

    train(model, data_loader_train, epochs=1)
    

if __name__ == "__main__":
    run()
