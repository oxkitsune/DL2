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
    samples["structure_masks"] = augment.augment_structure_masks(samples["structure_masks"])

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

# also i renamed the this function for clarity, but its the same as the train_model function apart from the teacher_forcing parameter
def ar_train_model(model, dataset, args, teacher_forcing=False):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device {device}")

    criterion = setup_loss(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
            model, train_dataloader, optimizer, criterion, teacher_forcing
        )
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
            batch["dose"].clone().detach(),
            batch["voxel_dimensions"].clone().detach(),
            batch["structure_masks"].clone().detach(),
        ),
    }

# basically only this function is different from the train_model function, but kept it seperate so @Gijs you can see 
# for yourself how to handle this
def train_single_epoch(model, data_loader, optimizer, criterion, teacher_forcing):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Train", leave=False)
    
    for batch in pbar:
        
        target = batch["dose"].unsqueeze(1)
        input_target_dose = torch.zeros_like(target, device=device)
        
        # ensure features/dose are in the correct shape
        # (batch_size, channels, height, width, depth)
        features = batch["features"].transpose(1, -1)
        
        combined_input = torch.cat([features, input_target_dose], dim=1)
        
        for x in range(0, 128, 8):
        
            optimizer.zero_grad()
            outputs = model(combined_input)
            loss = criterion(outputs, target[:, :, :, :, x:x+8])
            
            if teacher_forcing:
                combined_input[:, 3, :, :, x:x+8] = target[:, :, :, :, x:x+8]
            else:
                combined_input[:, 3, :, :, x:x+8] = outputs.detach()
            
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(data_loader)


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
            metrics["dose_score"] += dose_score(
                outputs, target, batch["possible_dose_mask"]
            ).item()
            metrics["mean_dvh_error"] += mean_dvh_error(
                outputs,
                batch["dose"].clone().detach(),
                batch["voxel_dimensions"].clone().detach(),
                batch["structure_masks"].clone().detach(),
            ).item()

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
