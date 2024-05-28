import torch
from tqdm import tqdm
import wandb
import os

from src.metrics import dose_score, mean_dvh_error
from src.data import Augment
from torch.utils.data import DataLoader, default_collate
from src.training.loss import RadiotherapyLoss

augment = Augment(42)


def transform(samples):
    samples = default_collate(samples)
    samples["features"] = augment.fit(samples["features"])
    samples["dose"] = augment.augment_dose(samples["dose"])
    samples["structure_masks"] = augment.augment_structure_masks(
        samples["structure_masks"]
    )

    return samples


def setup_loss(args):
    if args.loss == "all":
        return RadiotherapyLoss()
    elif args.loss == "mae":
        return RadiotherapyLoss(use_dvh=False, use_moment=False)
    elif args.loss == "dvh":
        return RadiotherapyLoss(use_moment=False)
    elif args.loss == "moment":
        return RadiotherapyLoss(use_dvh=False)

    raise Exception(f"{args.loss} is ot a valid loss function")


def train_model(model, dataset, args):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device {device}")

    criterion = setup_loss(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        collate_fn=transform,
        drop_last=True,
    )
    dev_data_loader = DataLoader(
        dataset["validation"], batch_size=args.batch_size, drop_last=True
    )

    pbar = tqdm(range(args.epochs), desc="Training model")
    for epoch in pbar:
        train_metrics = train_single_epoch(
            model, train_dataloader, optimizer, criterion
        )
        scheduler.step()
        dev_metrics = evaluate(model, dev_data_loader, criterion)

        wandb.log(
            {
                "train_loss": train_metrics["loss"],
                "train_dose_score": train_metrics["dose_score"],
                "train_mean_dvh_error": train_metrics["mean_dvh_error"],
                "dev_loss": dev_metrics["loss"],
                "dev_dose_score": dev_metrics["dose_score"],
                "dev_mean_dvh_error": dev_metrics["mean_dvh_error"],
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        save_model_checkpoint_for_epoch(model)

        pbar.write(f"============ Epoch {epoch}/{args.epochs} =============")
        pbar.write("Training metrics:")
        pbar.write(f"Loss {train_metrics['loss']:.3f}")
        pbar.write(f"Dose score {train_metrics['dose_score']:.3f}")
        pbar.write(f"Mean DVH error {train_metrics['mean_dvh_error']:.3f}")
        pbar.write("")
        pbar.write("Dev metrics:")
        pbar.write(f"Loss {dev_metrics['loss']:.3f}")
        pbar.write(f"Dose score {dev_metrics['dose_score']:.3f}")
        pbar.write(f"Mean DVH error {dev_metrics['mean_dvh_error']:.3f}")
        pbar.write("=======================================")


def compute_metrics(prediction, batch):
    return {
        "dose_score": dose_score(
            prediction, batch["dose"], batch["possible_dose_mask"]
        ),
        "mean_dvh_error": mean_dvh_error(
            prediction,
            batch["dose"],
            batch["voxel_dimensions"],
            batch["structure_masks"],
        ),
    }


def train_single_epoch(model, data_loader, optimizer, criterion):
    model.train()
    metrics = {"loss": 0, "dose_score": 0, "mean_dvh_error": 0}
    pbar = tqdm(data_loader, desc="Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        # ensure features/dose are in the correct shape
        # (batch_size, channels, height, width, depth)
        features = batch["features"].permute((0, 4, 1, 2, 3))

        # (batch_size, depth, width, height)
        target = batch["dose"].unsqueeze(1)
        structure_masks = batch["structure_masks"]

        outputs = model(features)

        loss = criterion(outputs, target, structure_masks)
        loss.backward()

        optimizer.step()
        metrics["loss"] += loss.item()

        batch_metrics = compute_metrics(outputs, batch)
        metrics["dose_score"] += batch_metrics["dose_score"].item()
        metrics["mean_dvh_error"] += batch_metrics["mean_dvh_error"].item()

    n_batches = len(data_loader)
    return {
        "loss": metrics["loss"] / n_batches,
        "dose_score": metrics["dose_score"] / n_batches,
        "mean_dvh_error": metrics["mean_dvh_error"] / n_batches,
    }


def evaluate(model, data_loader, criterion):
    model.eval()
    metrics = {"loss": 0, "dose_score": 0, "mean_dvh_error": 0}
    pbar = tqdm(data_loader, desc="Dev", leave=False)
    with torch.no_grad():
        for batch in pbar:
            features = batch["features"].permute((0, 4, 1, 2, 3))
            target = batch["dose"].unsqueeze(1)
            structure_masks = batch["structure_masks"]

            outputs = model(features)
            loss = criterion(outputs, target, structure_masks)

            metrics["loss"] += loss.item()

            batch_metrics = compute_metrics(outputs, batch)
            metrics["dose_score"] += batch_metrics["dose_score"].item()
            metrics["mean_dvh_error"] += batch_metrics["mean_dvh_error"].item()

    return {
        "loss": metrics["loss"] / len(data_loader),
        "dose_score": metrics["dose_score"] / len(data_loader),
        "mean_dvh_error": metrics["mean_dvh_error"] / len(data_loader),
    }


def save_model_checkpoint_for_epoch(model):
    # ensure checkpoints directory exists
    os.makedirs(".checkpoints", exist_ok=True)

    chpt_path = ".checkpoints/model_checkpoint.pt"

    # save model checkpoint
    torch.save(model.state_dict(), chpt_path)
    wandb.save(chpt_path)  # Do evaluation
