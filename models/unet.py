import torch
import torch.nn as nn
from torch import Tensor


class UNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()

        self.t_mlp = nn.Linear(1, base_channels)

        # (3, 28, 28) -> (32, 28, 28)
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        # (32, 28, 28) -> (64, 14, 14)
        self.encoder_1 = nn.Sequential(
            nn.GroupNorm(4, base_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
        )
        # (64, 14, 14) -> (128, 7, 7)
        self.encoder_2 = nn.Sequential(
            nn.GroupNorm(8, base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
        )
        # (128, 7, 7) -> (128, 7, 7)
        self.middle = nn.Sequential(
            nn.GroupNorm(16, base_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
        )
        # (128+128, 7, 7) -> (64, 14, 14)
        self.decoder_1 = nn.Sequential(
            nn.GroupNorm(32, base_channels * 8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(base_channels * 8, base_channels * 2, kernel_size=3, stride=1, padding=1),
        )
        # (64+64, 14, 14) -> (32, 28, 28)
        self.decoder_2 = nn.Sequential(
            nn.GroupNorm(16, base_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(base_channels * 4, base_channels, kernel_size=3, stride=1, padding=1),
        )
        # (32+32, 28, 28) -> (3, 28, 28)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        skips = []
        # first conv
        x = self.conv_in(x)
        t = self.t_mlp(t[:, None])
        x = x + t[:, :, None, None]
        skips.append(x)
        # encoder
        x = self.encoder_1(x)
        skips.append(x)
        x = self.encoder_2(x)
        skips.append(x)
        # middle
        x = self.middle(x)
        # decoder
        x = self.decoder_1(torch.cat([x, skips.pop(-1)], dim=1))
        x = self.decoder_2(torch.cat([x, skips.pop(-1)], dim=1))
        # last conv
        x = self.conv_out(torch.cat([x, skips.pop(-1)], dim=1))
        return x
