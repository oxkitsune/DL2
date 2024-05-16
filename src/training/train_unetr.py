import torch
from tqdm import tqdm
import wandb
import os


def train_unetr(data_loader, model, epochs, ptv_index):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0001
    )  # use lr 1e-4 and AdamW

    model.to(device)
    model.train()

    trange = tqdm(range(epochs), desc="Training model")

    for epoch in trange:
        data_loader.shuffle_data()

        subtrange = tqdm(
            data_loader.get_batches(), desc="Train", leave=False, total=len(data_loader)
        )

        total_loss = 0

        for batch in subtrange:
            features = batch.get_augmented_features(ptv_index=ptv_index)
            input = torch.Tensor(features).transpose(1, 4).to(device)
            target = batch.dose
            target = torch.Tensor(target).transpose(1, 4).to(device)

            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})
            total_loss += loss.item()

        trange.write(
            f"Model loss at epoch {epoch} is {(total_loss / len(data_loader)):.3f}"
        )
        save_model_checkpoint_for_epoch(model, epoch)


def save_model_checkpoint_for_epoch(model, epoch):
    # ensure checkpoints directory exists
    os.makedirs(".checkpoints", exist_ok=True)

    # save model checkpoint
    torch.save(model.state_dict(), f"checkpoints/model_checkpoint_epoch_{epoch}.pt")
    wandb.save(f".checkpoints/model_checkpoint_epoch_{epoch}.pt")
