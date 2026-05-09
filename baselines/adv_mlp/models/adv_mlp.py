"""Adv-MLP: adversarially regularized MLP for APS prediction."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Generator(nn.Module):
    """
    条件 Generator: cond_img → APS (180)

    架构: ResNet-18 encoder + MLP decoder
    """

    def __init__(self, seq_len=180, z_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.z_dim = z_dim

        # Condition encoder (ResNet-18)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.encoder = resnet  # 输出 (B, 512)

        # Noise projection
        self.noise_proj = nn.Linear(z_dim, 256)

        # Decoder: concat(cond_feat, noise) → APS
        self.decoder = nn.Sequential(
            nn.Linear(512 + 256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, seq_len),
        )

    def forward(self, cond_img, z=None):
        """
        Args:
            cond_img: (B, 3, H, W) condition image
            z: (B, z_dim) noise vector. If None, sample from N(0,1)

        Returns:
            aps_pred: (B, 180)
        """
        B = cond_img.shape[0]
        if z is None:
            z = torch.randn(B, self.z_dim, device=cond_img.device)

        cond_feat = self.encoder(cond_img)      # (B, 512)
        z_feat = self.noise_proj(z)             # (B, 256)
        feat = torch.cat([cond_feat, z_feat], dim=1)  # (B, 768)
        aps_pred = self.decoder(feat)           # (B, 180)
        return aps_pred


class Discriminator(nn.Module):
    """
    条件 Discriminator: (cond_img, aps) → real/fake

    架构: ResNet-18 处理 cond_img + MLP 处理 APS，concat 后判别
    """

    def __init__(self, seq_len=180):
        super().__init__()

        # Condition encoder (轻量版，不用预训练)
        resnet = resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.cond_encoder = resnet  # (B, 512)

        # APS encoder
        self.aps_encoder = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Discriminator head
        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, cond_img, aps):
        """
        Args:
            cond_img: (B, 3, H, W)
            aps: (B, 180)

        Returns:
            logits: (B, 1)
        """
        cond_feat = self.cond_encoder(cond_img)  # (B, 512)
        aps_feat = self.aps_encoder(aps)          # (B, 256)
        feat = torch.cat([cond_feat, aps_feat], dim=1)
        return self.head(feat)


class AdvMLP(nn.Module):
    """
    Adv-MLP wrapper: 封装 Generator 和 Discriminator
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config['model']
        seq_len = model_cfg['seq_len']
        z_dim = model_cfg.get('z_dim', 128)

        self.generator = Generator(seq_len=seq_len, z_dim=z_dim)
        self.discriminator = Discriminator(seq_len=seq_len)
        self.z_dim = z_dim
        self.seq_len = seq_len

    def forward(self, cond_img, z=None):
        """Generator forward (for inference)"""
        return self.generator(cond_img, z)
