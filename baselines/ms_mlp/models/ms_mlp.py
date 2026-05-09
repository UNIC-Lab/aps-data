"""MS-MLP: Multi-Scale Encoder + MLP Decoder, pure regression (no adversarial regularization).

Ablation counterpart of MS-AReg: same MultiScaleEncoder + same MLP Decoder,
but trained with MSE loss only. Isolates the contribution of multi-scale encoding
from adversarial regularization.

Architecture identical to MS-AReg Predictor, with z_dim=0 (no noise input).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MultiScaleEncoder(nn.Module):
    """ResNet-18 multi-scale feature extraction: fuses 4 stage outputs.
    Identical to MS-AReg's MultiScaleEncoder."""

    def __init__(self, proj_dim=256, pretrained=True):
        super().__init__()

        if pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet18(weights=None)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 64-ch
        self.layer2 = resnet.layer2   # 128-ch
        self.layer3 = resnet.layer3   # 256-ch
        self.layer4 = resnet.layer4   # 512-ch

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
        self.out_dim = proj_dim * 2

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

        return self.global_proj(torch.cat([s1, s2, s3, s4], dim=1))  # (B, proj_dim*2)


class MSMLP(nn.Module):
    """MS-MLP: Multi-Scale Encoder + MLP Decoder, no adversarial regularization.

    Ablation of MS-AReg: keeps multi-scale encoding, removes adversarial training.
    Pure deterministic regression; no noise input z.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        seq_len    = model_cfg['seq_len']
        proj_dim   = model_cfg.get('proj_dim', 256)
        pretrained = model_cfg.get('use_pretrained', True)

        self.encoder = MultiScaleEncoder(proj_dim=proj_dim, pretrained=pretrained)
        cond_dim = proj_dim * 2   # 512

        self.decoder = nn.Sequential(
            nn.Linear(cond_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, seq_len),
        )

    def forward(self, cond_img):
        feat = self.encoder(cond_img)   # (B, 512)
        return self.decoder(feat)       # (B, seq_len)
