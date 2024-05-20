import argparse
from pathlib import Path

import wandb

from src.data import transform_data
from src.models import UNETR

from src.training import train_unetr
from datasets import load_dataset, Array4D

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
        "--lr", type=float, default=1e-04, help="The learning rate for the model"
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

    parser.add_argument("--parallel", action="store_true", help="Use multiple GPUs")

    return parser.parse_args()


def setup_wandb(args):
    resume_id = None
    if args.resume_run:
        resume_id = args.resume_run.split("/")[-1]

    # start a new wandb run to track this script
    wandb.init(
        entity="jkbkaiser1",
        # set the wandb project where this run will be logged
        project=args.project,
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "Unetr",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        # this repo contains the entire dataset and code, so let's not upload it
        save_code=False,
        id=resume_id,
        # if this is a dry run, don't actually log anything
        mode="disabled" if args.dry_run else "online",
        resume=args.resume_run is not None,
    )


def run():
    args = get_args()
    setup_wandb(args)

    num_proc = torch.multiprocessing.cpu_count() - 2

    dataset = load_dataset("oxkitsune/open-kbp", num_proc=num_proc)

    # apply transformations in numpy format, on cpu
    dataset = (
        dataset.with_format("numpy")
        .map(
            transform_data,
            input_columns=["ct", "structure_masks"],
            # we remove these columns as they are combined into the 'features' column or irrelevant
            remove_columns=["ct", "structure_masks", "possible_dose_mask"],
            writer_batch_size=25,
            num_proc=num_proc,
        )
        # cast the features column to a 4D array, to make converting to torch 100x faster
        .cast_column("features", Array4D((128, 128, 128, 3), dtype="float32"))
    )

    dataset = dataset.with_format("torch", columns=["features", "dose"])
    # run the training loop
    train_unetr(dataset, args)


if __name__ == "__main__":
    run()
