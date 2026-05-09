"""RadioUNet - UNet-based direct regression for APS prediction"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RadioUNet(nn.Module):
    """
    RadioUNet for APS Prediction

    经典 2D UNet，输入 condition image (3, 256, 256)，
    通过 encoder-decoder 提取特征，最后用 MLP 回归 APS (180,)。

    架构:
        Encoder: 4 层下采样 (3→64→128→256→512)
        Bottleneck: 512→1024
        Decoder: 4 层上采样 (1024→512→256→128→64)
        Head: GlobalAvgPool → MLP → APS (180)
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config['model']
        self.seq_len = model_cfg['seq_len']

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Regression head: global pool → MLP → APS
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.seq_len),
        )

    def forward(self, cond_img):
        """
        Args:
            cond_img: (B, 3, 256, 256) condition image

        Returns:
            aps_pred: (B, 180) predicted APS
        """
        # Encoder
        e1 = self.enc1(cond_img)         # (B, 64, 256, 256)
        e2 = self.enc2(self.pool(e1))    # (B, 128, 128, 128)
        e3 = self.enc3(self.pool(e2))    # (B, 256, 64, 64)
        e4 = self.enc4(self.pool(e3))    # (B, 512, 32, 32)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, 16, 16)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))    # (B, 512, 32, 32)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # (B, 256, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # (B, 128, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # (B, 64, 256, 256)

        # Global pool → MLP regression
        feat = self.global_pool(d1).flatten(1)  # (B, 64)
        aps_pred = self.head(feat)               # (B, 180)

        return aps_pred
