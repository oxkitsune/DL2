import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import DataLoader


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain("relu")
        )
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain("relu")
        )


def train(
    model: nn.Module, data_loader: DataLoader, epochs: int, logger, ptv_index: int = 7
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loss_func = nn.L1Loss() # MAE
    optimizer = optim.Adam(model.parameters(), lr=1e-03)

    model.apply(weights_init)
    model = model.to(device)

    for _ in range(epochs):
        model.train()

        for batch in tqdm(data_loader.get_batches(), total=len(data_loader)):
            features = batch.get_all_features(ptv_index=ptv_index)
            input = torch.Tensor(features).transpose(1, 4).to(device)

            target = batch.get_target()
            target = torch.Tensor(target).transpose(1, 4).to(device)

            optimizer.zero_grad()

            output = model(input)
            loss = loss_func(output, target)
            logger.log({"loss": loss})

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pass
            # Do evaluation
