import torch
import torch.nn as nn
from torch import Tensor


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv3d(3, 8, 3, padding=1)
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv3_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv4_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv4_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv5_1 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv5_2 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv6_1 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv6_2 = nn.Conv3d(256, 256, 3, padding=1)
        self.output = nn.Conv3d(8, 1, 3, padding=1)

        self.convup1 = nn.Conv3d(256, 128, 3, padding=1)
        self.convup2 = nn.Conv3d(128, 64, 3, padding=1)
        self.convup3 = nn.Conv3d(64, 32, 3, padding=1)
        self.convup4 = nn.Conv3d(32, 16, 3, padding=1)
        self.convup5 = nn.Conv3d(16, 8, 3, padding=1)

        self.uptrans1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.uptrans2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.uptrans3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.uptrans4 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.uptrans5 = nn.ConvTranspose3d(16, 8, 2, stride=2)

        self.GN1 = nn.GroupNorm(1, 8)
        self.GN2 = nn.GroupNorm(2, 16)
        self.GN3 = nn.GroupNorm(4, 32)
        self.GN4 = nn.GroupNorm(8, 64)
        self.GN5 = nn.GroupNorm(16, 128)
        self.GN6 = nn.GroupNorm(32, 256)

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU()


    def forward(self, x: Tensor) -> Tensor:
        x1 = self.GN1(self.relu(self.conv1_2(self.GN1(self.relu(self.conv1_1(x))))))
        x = self.pool(x1)

        x2 = self.GN2(self.relu(self.conv2_2(self.GN2(self.relu(self.conv2_1(x))))))
        x = self.pool(x2)

        x3 = self.GN3(self.relu(self.conv3_2(self.GN3(self.relu(self.conv3_1(x))))))
        x = self.pool(x3)

        x4 = self.GN4(self.relu(self.conv4_2(self.GN4(self.relu(self.conv4_1(x))))))
        x = self.pool(x4)

        x5 = self.GN5(self.relu(self.conv5_2(self.GN5(self.relu(self.conv5_1(x))))))
        x = self.pool(x5)

        x = self.GN6(self.relu(self.conv6_2(self.GN6(self.relu(self.conv6_1(x))))))

        x = self.GN5(self.relu(self.uptrans1(x)))
        x = torch.cat((x, x5), dim=1)
        x = self.GN5(self.relu(self.conv5_2(self.GN5(self.relu(self.convup1(x))))))
        del x5

        x = self.GN4(self.relu(self.uptrans2(x)))
        x = torch.cat((x, x4), dim=1)
        x = self.GN4(self.relu(self.conv4_2(self.GN4(self.relu(self.convup2(x))))))
        del x4

        x = self.GN3(self.relu(self.uptrans3(x)))
        x = torch.cat((x, x3), dim=1)
        x = self.GN3(self.relu(self.conv3_2(self.GN3(self.relu(self.convup3(x))))))
        del x3

        x = self.GN2(self.relu(self.uptrans4(x)))
        x = torch.cat((x, x2), dim=1)
        x = self.GN2(self.relu(self.conv2_2(self.GN2(self.relu(self.convup4(x))))))
        del x2

        x = self.GN1(self.relu(self.uptrans5(x)))
        x = torch.cat((x, x1), dim=1)
        x = self.GN1(self.relu(self.conv1_2(self.GN1(self.relu(self.convup5(x))))))

        x = self.relu(self.output(x))
        return x
