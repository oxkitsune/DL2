import torch
from tqdm import tqdm
import wandb
import os

from src.models.unetr import UNETR
from src.models.u_net import TrDosePred
from torch.utils.data import DataLoader


def train_unetr(dataset, args):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device {device}")

    model = UNETR(input_dim=3, output_dim=1).to(device)
    # model = TrDosePred().to(device)

    if args.parallel:
        model = torch.nn.DataParallel(model, output_device=device)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size)
    dev_data_loader = DataLoader(dataset["validation"], batch_size=args.batch_size)

    pbar = tqdm(range(args.epochs), desc="Training model")
    for epoch in pbar:
        train_loss = train_single_epoch(model, train_dataloader, optimizer, criterion)
        dev_loss = evaluate(model, dev_data_loader, criterion)

        wandb.log({"train_loss": train_loss, "dev_loss": dev_loss})
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
        features = batch["features"].transpose(1, -1)
        target = batch["dose"].unsqueeze(1)
        outputs = model(features)

        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Dev", leave=False)
    for batch in pbar:
        features = batch["features"].transpose(1, -1)
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
    wandb.save(chpt_path)

if __name__ == "__main__":
    import numpy as np
    NUM_RELEVANT_STRUCTURES = 7
    def transform_data(ct, structure_masks):
        # Need to add a channel dimension to the CT data
        # in order to end up with a feature map of shape (128, 128, 128, 3)
        ct_channel = np.expand_dims(ct, -1)

        # These are all encoded as ones and zeros originally
        oar_channels = structure_masks[..., :NUM_RELEVANT_STRUCTURES]
        labels = np.arange(1, NUM_RELEVANT_STRUCTURES + 1)
        labeled_oar_channels = oar_channels * labels
        oar_channel = np.sum(labeled_oar_channels, axis=-1, keepdims=True)

        # According to the original paper, the PTV channel is normalized by 70
        ptv_channels = structure_masks[..., NUM_RELEVANT_STRUCTURES:]
        labels = np.array([56, 63, 70])
        ptv_channel = np.max(ptv_channels * labels, axis=-1, keepdims=True) / 70

        # Combine the channels into a single feature tensor
        flattened_features = np.concatenate((ct_channel, oar_channel, ptv_channel), axis=-1)

        return {
            "features": flattened_features,
        }

    from datasets import load_dataset, Array4D
    dataset = load_dataset("oxkitsune/open-kbp", num_proc=1)

    # apply transformations in numpy format, on cpu
    dataset = (
        dataset.with_format("numpy")
        .map(
            transform_data,
            input_columns=["ct", "structure_masks"],
            # we remove these columns as they are combined into the 'features' column or irrelevant
            remove_columns=["ct", "structure_masks", "possible_dose_mask"],
            writer_batch_size=25,
            num_proc=1,
        )
        # cast the features column to a 4D array, to make converting to torch 100x faster
        .cast_column("features", Array4D((128, 128, 128, 3), dtype="float32"))
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = dataset.with_format("torch", columns=["features", "dose"], device=device)

    model = TrDosePred(n_heads = 1)
    data_loader = DataLoader(dataset["train"], batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()
    train_single_epoch(model, data_loader, optimizer, criterion)