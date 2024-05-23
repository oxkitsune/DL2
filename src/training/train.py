import torch
from tqdm import tqdm
import wandb
import os


from src.data import Augment
from torch.utils.data import DataLoader, default_collate
from src.training.loss import RadiotherapyLoss

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

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    criterion = RadiotherapyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = DataLoader(
        dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=transform
    )
    dev_data_loader = DataLoader(dataset["validation"], batch_size=args.batch_size)

    pbar = tqdm(range(args.epochs), desc="Training model")
    for epoch in pbar:
        train_loss = train_single_epoch(model, train_dataloader, optimizer, criterion)
        dev_loss = evaluate(model, dev_data_loader, criterion)

        wandb.log({"train_loss": train_loss, "dev_loss": dev_loss, "epoch": epoch})
        save_model_checkpoint_for_epoch(model)

        pbar.write(
            f"[{epoch}/{args.epochs}] Train loss: {train_loss:.3f} Dev loss: {dev_loss:.3f}"
        )


def train_single_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0    
    pbar = tqdm(data_loader, desc="Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        # ensure features/dose are in the correct shape
        # (batch_size, channels, height, width, depth)
        features = batch["features"].permute((0, 4, 1, 2, 3))
        target = batch["dose"].unsqueeze(1) / 70
        structure_masks = batch["structure_masks"]

        outputs = model(features)

        loss = criterion(outputs, target, structure_masks)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Dev", leave=False)
    with torch.no_grad():
        for batch in pbar:
            features = batch["features"].permute((0, 4, 1, 2, 3))
            target = batch["dose"].unsqueeze(1)
            structure_masks = batch["structure_masks"]

            outputs = model(features)

            loss = criterion(outputs, target, structure_masks)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def save_model_checkpoint_for_epoch(model):
    # ensure checkpoints directory exists
    os.makedirs(".checkpoints", exist_ok=True)

    chpt_path = ".checkpoints/model_checkpoint.pt"

    # save model checkpoint
    torch.save(model.state_dict(), chpt_path)
    wandb.save(chpt_path)  # Do evaluation
