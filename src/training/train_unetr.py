import torch
from tqdm import tqdm
import wandb
import os

from src.evaluation import evaluate


def train_unetr(data_loader, model, epochs, data_loader_validation, PREDICTION_DIR):
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
        model.train()
        data_loader.shuffle_data()

        subtrange = tqdm(
            data_loader.get_batches(),
            desc="Train",
            leave=False,
            total=len(data_loader) // data_loader.batch_size,
        )

        total_loss = 0

        for batch in subtrange:
            features = batch.get_augmented_features()
            input = torch.Tensor(features).transpose(1, 4).to(device)
            target = batch.dose
            target = torch.Tensor(target).transpose(1, 4).to(device)

            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            wandb.log({"loss_per_batch": loss.item()})
            total_loss += loss.item()

        loss_for_epoch = total_loss / len(data_loader)

        save_model_checkpoint_for_epoch(model, epoch)

        model.eval()
        dvh_score, dose_score = evaluate(model, data_loader_validation, PREDICTION_DIR)
        wandb.log(
            {
                "dev_dvh_score": dvh_score,
                "dev_dose_score": dose_score,
                "mean_loss_per_epoch": loss_for_epoch,
            }
        )
        trange.write(f"Train loss at epoch {epoch} is {loss_for_epoch:.3f}")
        trange.write(f"Validation DVH score: {dvh_score:.3f}")
        trange.write(f"Validation dose score: {dose_score:.3f}")
        trange.write("=====================================")


def save_model_checkpoint_for_epoch(model, epoch):
    # ensure checkpoints directory exists
    os.makedirs(".checkpoints", exist_ok=True)

    chpt_path = f".checkpoints/model_checkpoint_epoch.pt"

    # save model checkpoint
    torch.save(model.state_dict(), chpt_path)
    wandb.save(chpt_path)
