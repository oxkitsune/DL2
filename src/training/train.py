import torch
from tqdm import tqdm
import wandb
import os

from src.metrics import dose_score, mean_dvh_error
from src.data import Augment
from torch.utils.data import DataLoader, default_collate

augment = Augment(42)


def transform(samples):
    samples = default_collate(samples)
    samples["features"] = augment(samples["features"])
    samples["dose"] = augment(samples["dose"])

    return samples


def train_model(model, dataset, args):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device {device}")

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = DataLoader(
        dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=transform
    )
    dev_data_loader = DataLoader(dataset["validation"], batch_size=args.batch_size)

    pbar = tqdm(range(args.epochs), desc="Training model")
    for epoch in pbar:
        train_metrics = train_single_epoch(
            model, train_dataloader, optimizer, criterion
        )
        print(train_metrics)
        dev_loss = evaluate(model, dev_data_loader, criterion)

        wandb.log(
            {
                "train_loss": train_metrics["loss"],
                "train_dose_score": train_metrics["dose_score"],
                "train_mean_dvh_error": train_metrics["mean_dvh_error"],
                "dev_loss": dev_loss,
                "epoch": epoch,
            }
        )
        save_model_checkpoint_for_epoch(model)

        pbar.write(
            f"[{epoch}/{args.epochs}] Train loss: {train_metrics['loss']:.3f} Dev loss: {dev_loss:.3f}"
        )


def compute_metrics(prediction, batch):
    return {
        "dose_score": dose_score(
            prediction, batch["dose"], batch["possible_dose_mask"]
        ),
        "mean_dvh_error": mean_dvh_error(prediction, batch),
    }


def train_single_epoch(model, data_loader, optimizer, criterion):
    model.train()
    metrics = {"loss": 0, "dose_score": 0, "mean_dvh_error": 0}
    pbar = tqdm(data_loader, desc="Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        # ensure features/dose are in the correct shape
        # (batch_size, channels, depth, width, height)
        features = batch["features"].permute((0, 4, 1, 2, 3))

        # (batch_size, depth, width, height)
        target = batch["dose"].unsqueeze(1)

        outputs = model(features)

        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()
        metrics["loss"] += loss.item()
        metrics["dose_score"] += dose_score(
            outputs, target, batch["possible_dose_mask"]
        )
        metrics["mean_dvh_error"] += mean_dvh_error(outputs, batch)

    n_batches = len(data_loader)
    return {
        "loss": metrics["loss"] / n_batches,
        "dose_score": metrics["dose_score"] / n_batches,
        "mean_dvh_error": metrics["mean_dvh_error"] / n_batches,
    }


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Dev", leave=False)
    with torch.no_grad():
        for batch in pbar:
            features = batch["features"].permute((0, 4, 1, 2, 3))
            target = batch["dose"].unsqueeze(1)

            outputs = model(features)

            loss = criterion(outputs, target)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def save_model_checkpoint_for_epoch(model):
    # ensure checkpoints directory exists
    os.makedirs(".checkpoints", exist_ok=True)

    chpt_path = ".checkpoints/model_checkpoint.pt"

    # save model checkpoint
    torch.save(model.state_dict(), chpt_path)
    wandb.save(chpt_path)  # Do evaluation
