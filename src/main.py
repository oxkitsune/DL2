from pathlib import Path
from typing import Optional, TypeVar

import wandb

from src.data import DataLoader, get_paths
from src.models import UNETR
from src.training import train_unetr
from src.evaluation import evaluate

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

PREDICTION_DIR = Path("results")
# Original paper
# EPOCHS = 200
EPOCHS = 25

def run():
    primary_directory = Path().resolve()
    provided_data_dir = primary_directory / "data"
    training_data_dir = provided_data_dir / "train-pats"
    validation_data_dir = provided_data_dir / "validation-pats"
    training_plan_paths = get_paths(training_data_dir)
    validation_plan_paths = get_paths(validation_data_dir)
    data_loader_train = DataLoader(training_plan_paths)
    data_loader_validation = DataLoader(validation_plan_paths)

    data_loader_train.set_mode("training_model")
    data_loader_validation.set_mode("training_model")

    model = UNETR(input_dim=3, output_dim=1)
    train_unetr(data_loader_train, model, EPOCHS)
    res = evaluate(model, data_loader_validation, PREDICTION_DIR)
    print(res)


if __name__ == "__main__":
    run()
