import argparse

from pathlib import Path
from typing import Optional, TypeVar

import wandb

from src.data import DataLoader, get_paths
from src.models import UNETR
from src.training import train_unetr
from src.evaluation import evaluate

import torch

PREDICTION_DIR = Path("results")
# Original paper
# EPOCHS = 200
# EPOCHS = 25


def get_args():
    parser = argparse.ArgumentParser(description="Train a model on the OpenKBP dataset")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="The number of epochs to train the model for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The number of data points to load in a single batch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for the random number generator",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script without wandb logging",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="dl2",
        help="The wandb project to log this run to",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-04, help="The learning rate for the model"
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        default=None,
        help="The ID of a wandb run to resume",
    )
    parser.add_argument(
        "--restore-checkpoint",
        type=str,
        default=None,
        help="The path to a model checkpoint to restore",
    )

    return parser.parse_args()


def setup_wandb(args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.project,
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "Unetr",
            "epochs": args.epochs,
        },
        # this repo contains the entire dataset and code, so let's not upload it
        save_code=False,
        # if this is a dry run, don't actually log anything
        mode="disabled" if args.dry_run else "online",
        id=args.resume_run,
        resume=args.resume_run is not None,
    )


def run():
    args = get_args()

    setup_wandb(args)

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UNETR(input_dim=3, output_dim=1)
    if args.resume_run:
        run = wandb.Api().run(args.resume_run)
        print(f"Loading model checkpoint {args.restore_checkpoint}")
        run.file(args.restore_checkpoint).download(exist_ok=True)
        checkpoint = torch.load(
            args.restore_checkpoint, weights_only=True, map_location=device
        )
        model.load_state_dict(checkpoint)

    train_unetr(
        data_loader_train, model, args.epochs, data_loader_validation, PREDICTION_DIR
    )


if __name__ == "__main__":
    run()
