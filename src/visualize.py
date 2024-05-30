import argparse
import datetime
import subprocess
import torch
import wandb

from pathlib import Path

from datasets import load_dataset, Array4D
import numpy as np
from torch.utils.data import DataLoader, default_collate
import matplotlib.pyplot as plt

from src.data import transform_data
from src.training import train_model, ar_train_model
from src.training.train import load_model_weights


def get_args():
    parser = argparse.ArgumentParser(description="Script to visualize predictions")
    parser.add_argument(
        "run",
        type=str,
        help="The ID of a wandb run to resume",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unetr",
        help="The model to use for training",
    )

    return parser.parse_args()


def setup_model(args, device):
    if args.model == "unetr":
        from src.models import UNETR

        model = UNETR(input_dim=3, output_dim=1).to(device)
    elif args.model == "bigunetr":
        from src.models import UNETRBig

        model = UNETRBig(input_dim=3, output_dim=1).to(device)
    elif args.model == "conv":
        from src.models import ConvNet

        model = ConvNet(num_input_channels=3).to(device)
    elif args.model == "arunetr":
        from src.models import AR_UNETR

        model = AR_UNETR(input_dim=4, output_dim=1).to(device)
    elif args.model == "rnnunetr":
        from src.models import UNETR_RNN

        model = UNETR_RNN(input_dim=3, output_dim=1).to(device)
    else:
        raise ValueError(f"Unknown model {args.model}")

    # run = wandb.Api().run(args.run)
    # print(f"Downloading model checkpoint {args.run}...")
    # run.file(".checkpoints/model_checkpoint.pt").download(replace=True)

    return model


def run():
    args = get_args()
    num_proc = 2
    # num_proc = torch.multiprocessing.cpu_count() - 2
    dataset = load_dataset("oxkitsune/open-kbp", num_proc=num_proc)

    # ensure the feature format is set for the new features column, this speeds up the dataset loading by 100x
    features = dataset["train"].features.copy()
    features["features"] = Array4D((128, 128, 128, 3), dtype="float32")
    del features["ct"]

    # apply transformations in numpy format, on cpu
    dataset = dataset.with_format("numpy").map(
        transform_data,
        input_columns=["ct", "structure_masks"],
        # we remove these columns as they are combined into the 'features' column or irrelevant
        remove_columns=["ct"],
        writer_batch_size=25,
        num_proc=num_proc,
        features=features,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = dataset.with_format(
        "torch",
        columns=[
            "features",
            "dose",
            "structure_masks",
            "voxel_dimensions",
            "possible_dose_mask",
        ],
        device=device,
    )
    model = setup_model(args, device)

    # Load model weights
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device {device}")

    # checkpoint_path = Path(".checkpoints/model_checkpoint.pt")
    # load_model_weights(model, device, checkpoint_path)

    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=1,
        drop_last=True,
    )

    model.eval()
    metrics = {"dose_score": 0, "mean_dvh_error": 0}
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        features = batch["features"].permute((0, 4, 1, 2, 3))
        target = batch["dose"].unsqueeze(1)
        structure_masks = batch["structure_masks"]
        outputs = model(features)

    create_gifs(np.squeeze(outputs.cpu()[0]))


def create_gifs(data):
    for i, axis in enumerate(["axial", "sagital", "coronal"]):
        subprocess.run(
            ["rm", "-rf", "*"],
            cwd="./figs/gifs",
        )

        for j in range(data.shape[-(i + 1)]):
            plt.axis("off")

            if i == 0:
                plt.imshow(data[:, :, j])
            elif i == 1:
                plt.imshow(data[:, j, :])
            elif i == 2:
                plt.imshow(data[j, :, :])

            plt.savefig(f"./figs/gifs/frame{j}.png")

        subprocess.run(
            [
                "ffmpeg",
                "-framerate",
                "16",
                "-i",
                "frame%d.png",
                "-vf",
                "scale=iw:-1:flags=lanczos",
                "-y",
                f"./../{axis}.gif",
            ],
            cwd="./figs/gifs",
        )


if __name__ == "__main__":
    run()
