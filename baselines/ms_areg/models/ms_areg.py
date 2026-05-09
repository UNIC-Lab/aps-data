"""MS-AReg: Multi-Scale Adversarial Regression for APS Prediction (Proposed Method)

Regression network with multi-scale feature extraction and adversarial regularization.
The regularizer is only used during training; inference is pure deterministic regression.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# Multi-Scale ResNet-18 Encoder
# ============================================================

class MultiScaleEncoder(nn.Module):
    """ResNet-18 multi-scale feature extraction: fuses 4 stage outputs."""

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
        self.proj_dim = proj_dim

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
        return global_cond  # (B, proj_dim*2)


# ============================================================
# Predictor (Multi-Scale Encoder + MLP Decoder)
# ============================================================

class Predictor(nn.Module):
    """
    Multi-Scale Regression Predictor: cond_img -> APS (180)

    Multi-scale ResNet-18 encoder fuses 4-stage features,
    MLP decoder maps to APS prediction.
    """

    def __init__(self, seq_len=180, z_dim=128, proj_dim=256, pretrained=True):
        super().__init__()
        self.seq_len = seq_len
        self.z_dim = z_dim

        self.encoder = MultiScaleEncoder(proj_dim=proj_dim, pretrained=pretrained)
        self.noise_proj = nn.Linear(z_dim, 256)

        self.decoder = nn.Sequential(
            nn.Linear(proj_dim * 2 + 256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, seq_len),
        )

    def forward(self, cond_img, z=None):
        B = cond_img.shape[0]
        if z is None:
            z = torch.zeros(B, self.z_dim, device=cond_img.device)

        global_cond = self.encoder(cond_img)    # (B, 512)
        z_feat = self.noise_proj(z)             # (B, 256)
        feat = torch.cat([global_cond, z_feat], dim=1)  # (B, 768)
        return self.decoder(feat)               # (B, 180)


# ============================================================
# Regularizer (adversarial, training only)
# ============================================================

class Regularizer(nn.Module):
    """Adversarial regularizer: scores (cond_img, aps) pairs. Training only."""

    def __init__(self, seq_len=180):
        super().__init__()

        resnet = resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.cond_encoder = resnet

        self.aps_encoder = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, cond_img, aps):
        cond_feat = self.cond_encoder(cond_img)
        aps_feat = self.aps_encoder(aps)
        feat = torch.cat([cond_feat, aps_feat], dim=1)
        return self.head(feat)


# ============================================================
# Wrapper
# ============================================================

class MSAReg(nn.Module):
    """MS-AReg: Multi-Scale Adversarial Regression"""

    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        seq_len = model_cfg['seq_len']
        z_dim = model_cfg.get('z_dim', 128)
        proj_dim = model_cfg.get('proj_dim', 256)
        pretrained = model_cfg.get('use_pretrained', True)

        self.predictor = Predictor(seq_len=seq_len, z_dim=z_dim,
                                   proj_dim=proj_dim, pretrained=pretrained)
        self.regularizer = Regularizer(seq_len=seq_len)
        self.z_dim = z_dim
        self.seq_len = seq_len

    def forward(self, cond_img, z=None):
        """Predictor forward (for inference)"""
        return self.predictor(cond_img, z)
