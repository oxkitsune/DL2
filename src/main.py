from pathlib import Path
from typing import Optional, TypeVar

import wandb

from src.data import DataLoader, get_paths
from src.models import UNETR
from src.training import train_unetr

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dl2",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-04,
        "architecture": "Unetr",
        "epochs": 1,
    },
)

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
    model = UNETR()

    ptv_index = 7
    model = UNETR(input_dim=3, output_dim=1)
    # epochs = 200 # original paper
    epochs = 25
    train_unetr(data_loader_train, model, epochs, ptv_index)


if __name__ == "__main__":
    run()
