"""ResNet-Conv1D baseline for direct APS regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FiLM(nn.Module):
    """Feature-wise linear modulation: gamma * x + beta."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class MultiScaleEncoder(nn.Module):
    """ResNet-18 multi-scale feature extractor used by ResNet-Conv1D."""

    def __init__(self, proj_dim=256, pretrained=True):
        super().__init__()

        if pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet18(weights=None)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.pool1 = nn.AdaptiveAvgPool2d(4)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)

        self.proj1 = nn.Linear(64 * 16, proj_dim)
        self.proj2 = nn.Linear(128 * 4, proj_dim)
        self.proj3 = nn.Linear(256, proj_dim)
        self.proj4 = nn.Linear(512, proj_dim)

        self.global_proj = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim * 2, proj_dim * 2),
        )

    def forward(self, x):
        h = self.stem(x)
        f1 = self.layer1(h)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        s1 = self.proj1(self.pool1(f1).flatten(1))
        s2 = self.proj2(self.pool2(f2).flatten(1))
        s3 = self.proj3(self.pool3(f3).flatten(1))
        s4 = self.proj4(self.pool4(f4).flatten(1))

        global_cond = self.global_proj(torch.cat([s1, s2, s3, s4], dim=1))
        return global_cond, [s3, s2, s1]


class ResNetConv1D(nn.Module):
    """
    Multi-scale encoder with a 1D convolutional APS decoder.
    用纯 MSE 回归训练。

    论文用途：隔离 "架构改进" 和 "对抗训练" 的贡献。
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        self.seq_len = model_cfg['seq_len']
        proj_dim = model_cfg.get('proj_dim', 256)
        pretrained = model_cfg.get('use_pretrained', True)

        # Multi-scale condition encoder.
        self.encoder = MultiScaleEncoder(proj_dim=proj_dim, pretrained=pretrained)

        # Initial projection: global_cond -> (B, 256, 45)  (无噪声)
        self.init_proj = nn.Linear(proj_dim * 2, 256 * 45)

        # 1D Conv decoder with FiLM conditioning.
        self.up1_conv1 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.up1_bn1 = nn.BatchNorm1d(128)
        self.up1_film = FiLM(proj_dim, 128)
        self.up1_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.up1_bn2 = nn.BatchNorm1d(128)

        self.up2_conv1 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.up2_bn1 = nn.BatchNorm1d(64)
        self.up2_film = FiLM(proj_dim, 64)
        self.up2_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.up2_bn2 = nn.BatchNorm1d(64)

        self.ref_conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.ref_bn = nn.BatchNorm1d(32)
        self.ref_film = FiLM(proj_dim, 32)
        self.ref_conv2 = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, cond_img):
        """
        Args:
            cond_img: (B, 3, H, W)
        Returns:
            aps_pred: (B, 180)
        """
        B = cond_img.shape[0]

        global_cond, scale_feats = self.encoder(cond_img)

        h = self.init_proj(global_cond)  # (B, 256*45)
        h = h.view(B, 256, 45)

        # Stage 1: 45 -> 90
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = F.leaky_relu(self.up1_bn1(self.up1_conv1(h)), 0.2)
        h = self.up1_film(h, scale_feats[0])
        h = F.leaky_relu(self.up1_bn2(self.up1_conv2(h)), 0.2)

        # Stage 2: 90 -> 180
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = F.leaky_relu(self.up2_bn1(self.up2_conv1(h)), 0.2)
        h = self.up2_film(h, scale_feats[1])
        h = F.leaky_relu(self.up2_bn2(self.up2_conv2(h)), 0.2)

        # Stage 3: refinement
        h = F.leaky_relu(self.ref_bn(self.ref_conv1(h)), 0.2)
        h = self.ref_film(h, scale_feats[2])
        h = self.ref_conv2(h)  # (B, 1, 180)

        return h.squeeze(1)
