import torch
from torch import nn
from Swin3D.modules.swin3d_layers import BasicLayer


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class EmbeddingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(3)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule1 = EmbeddingBlock()
        self.submodule2 = EmbeddingBlock()
        self.submodule3 = EmbeddingBlock()
        self.pointwise_conv = nn.Conv3d(in_channels=3, out_channels=96, kernel_size=1)
        self.downsampling = nn.Conv3d(
            in_channels=96, out_channels=96, kernel_size=(2, 4, 4), stride=(2, 4, 4)
        )

    def forward(self, x):
        x = self.submodule1(x)
        x = self.submodule2(x)
        x = self.submodule3(x)
        x = self.pointwise_conv(x)
        x = self.downsampling(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transpose_conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(2, 4, 4), stride=(2, 4, 4)
        )
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.transpose_conv(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_expanding_block = ExpandingBlock(in_channels=96, out_channels=96)
        self.pointwise_conv = nn.Conv3d(in_channels=96, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.patch_expanding_block(x)
        x = self.pointwise_conv(x)
        return x


class WrappedEncoder(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.encoder = BasicLayer(dim=dim, depth=depth)
        self.downsampling = Downsampling(in_channels=dim, out_channels=2 * dim)

    def forward(self, x):
        residual = self.encoder(x)
        x = self.downsampling(residual)
        return x, residual


class WrappedDecoder(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.decoder = BasicLayer(dim=dim, depth=depth)
        self.transpose_conv = nn.ConvTranspose3d(
            in_channels=dim, out_channels=dim // 2, kernel_size=2
        )

    def forward(self, x, residual):
        x += residual
        x = self.decoder(x)
        x = self.transpose_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO param values of basiclayer?
        self.encoder1 = WrappedEncoder(dim=96, depth=2)
        self.encoder2 = WrappedEncoder(dim=2 * 96, depth=2)
        self.encoder3 = WrappedEncoder(dim=4 * 96, depth=2)
        self.encoder4 = WrappedEncoder(dim=8 * 96, depth=1)

        self.decoder1 = WrappedDecoder(dim=8 * 96, depth=2)
        self.decoder2 = WrappedDecoder(dim=4 * 96, depth=2)
        self.decoder3 = WrappedDecoder(dim=2 * 96, depth=2)
        self.decoder4 = WrappedDecoder(dim=96, depth=2)

        # self.encoder1 = BasicLayer(dim=96, depth=2)
        # self.downsampling1 = Downsampling(in_channels=96, out_channels=2 * 96)
        # self.encoder2 = BasicLayer(dim=2 * 96, depth=2)
        # self.downsampling2 = Downsampling(in_channels=2 * 96, out_channels=4 * 96)
        # self.encoder3 = BasicLayer(dim=4 * 96, depth=2)
        # self.downsampling3 = Downsampling(in_channels=4 * 96, out_channels=8 * 96)
        # self.encoder4 = BasicLayer(dim=8 * 96, depth=1)

        # self.decoder1 = BasicLayer(dim=8 * 96, depth=2)
        # self.transpose_conv = nn.ConvTranspose3d(
        #     in_channels=8 * 96, out_channels=4 * 96, kernel_size=2
        # )
        # self.decoder2 = BasicLayer(dim=4 * 96, depth=2)
        # self.transpose_conv = nn.ConvTranspose3d(
        #     in_channels=4 * 96, out_channels=2 * 96, kernel_size=2
        # )
        # self.decoder3 = BasicLayer(dim=2 * 96, depth=2)
        # self.transpose_conv = nn.ConvTranspose3d(
        #     in_channels=2 * 96, out_channels=96, kernel_size=2
        # )
        # self.decoder4 = BasicLayer(dim=96, depth=2)

    def forward(self, x):
        # TODO residual connections
        x, residual1 = self.encoder1(x)
        x, residual2 = self.encoder2(x)
        x, residual3 = self.encoder3(x)
        x, residual4 = self.encoder4(x)

        x = self.decoder1(x, residual4)
        x = self.decoder2(x, residual3)
        x = self.decoder3(x, residual2)
        x = self.decoder4(x, residual1)

        # x1 = self.encoder1(x)
        # x1 = self.downsampling1(x1)
        # x2 = self.encoder2(x1)
        # x2 = self.downsampling2(x2)
        # x3 = self.encoder3(x2)
        # x3 = self.downsampling3(x3)
        # x4 = self.encoder4(x3)

        # y1 = self.decoder1(x4)
        # y1 = self.transpose_conv(y1)
        # y2 = self.decoder2(y1)
        # y2 = self.transpose_conv(y2)
        # y3 = self.decoder3(y2)
        # y3 = self.transpose_conv(y3)
        # y4 = self.decoder4(y3)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128, 128)
    embedding = PatchEmbedding()
    unet = UNet()
    expanding = PatchExpanding()

    y = embedding(x)
    print(y.shape)
    y = unet(y)
    print(y.shape)
    y = expanding(y)
    print(y.shape)
