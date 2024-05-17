import torch
from tqdm import tqdm
import wandb
import os

from src.models.unetr import UNETR
# from src.evaluation import evaluate

from torch.utils.data import DataLoader


def train_unetr(dataset, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")

    model = UNETR(input_dim=3, output_dim=1).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = DataLoader(
        dataset["train"], batch_size=args.batch_size, shuffle=True
    )
    dev_data_loader = DataLoader(
        dataset["validation"], batch_size=args.batch_size, shuffle=False
    )

    pbar = tqdm(range(args.epochs), desc="Training model")
    for epoch in pbar:
        train_loss = train_single_epoch(model, train_dataloader, optimizer, criterion)
        dev_loss = evaluate(model, dev_data_loader, criterion)

        pbar.write(
            f"[{epoch}/{args.epochs}] Train loss: {train_loss:.3f} Dev loss: {dev_loss:.3f}"
        )


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Eval", leave=False)
    for batch in pbar:
        features = batch["features"].transpose(1, -1)
        target = batch["dose"].unsqueeze(1)

        outputs = model(features)

        loss = criterion(outputs, target)
        total_loss += loss.item()

    return total_loss / len(data_loader)


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


# def train_unetr(dataset, model, epochs, data_loader_validation, PREDICTION_DIR):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     print(f"Using device {device}")
#     criterion = torch.nn.L1Loss()
#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=0.0001
#     )  # use lr 1e-4 and AdamW

#     model.to(device)
#     model.train()
#     trange = tqdm(range(epochs), desc="Training model")

#     for epoch in trange:
#         model.train()
#         data_loader.shuffle_data()

#         subtrange = tqdm(
#             data_loader.get_batches(),
#             desc="Train",
#             leave=False,
#             total=len(data_loader) // data_loader.batch_size,
#         )

#         total_loss = 0

#         for batch in subtrange:
#             features = batch.get_flattend_oar_features()
#             input = torch.Tensor(features).transpose(1, 4).to(device)
#             target = batch.dose
#             target = torch.Tensor(target).transpose(1, 4).to(device)

#             optimizer.zero_grad()
#             outputs = model(input)
#             loss = criterion(outputs, target)
#             loss.backward()
#             optimizer.step()
#             wandb.log({"loss_per_batch": loss.item()})
#             total_loss += loss.item()

#         loss_for_epoch = total_loss / len(data_loader)

#         save_model_checkpoint_for_epoch(model, epoch)

#         model.eval()
#         dvh_score, dose_score = evaluate(model, data_loader_validation, PREDICTION_DIR)
#         wandb.log(
#             {
#                 "dev_dvh_score": dvh_score,
#                 "dev_dose_score": dose_score,
#                 "mean_loss_per_epoch": loss_for_epoch,
#             }
#         )
#         trange.write(f"Train loss at epoch {epoch} is {loss_for_epoch:.3f}")
#         trange.write(f"Validation DVH score: {dvh_score:.3f}")
#         trange.write(f"Validation dose score: {dose_score:.3f}")
#         trange.write("=====================================")


def save_model_checkpoint_for_epoch(model, epoch):
    # ensure checkpoints directory exists
    os.makedirs(".checkpoints", exist_ok=True)

    chpt_path = f".checkpoints/model_checkpoint_epoch.pt"

    # save model checkpoint
    torch.save(model.state_dict(), chpt_path)
    wandb.save(chpt_path)
