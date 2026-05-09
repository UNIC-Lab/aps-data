"""ResNet-MLP - Simplest deep learning baseline for APS prediction"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetMLP(nn.Module):
    """
    ResNet-18 (pretrained) + 2-layer MLP head

    最简单的深度学习基线：提取全局特征后直接回归 APS。
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config['model']
        seq_len = model_cfg['seq_len']
        use_pretrained = model_cfg.get('use_pretrained', True)

        if use_pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.encoder = resnet  # -> (B, 512)

        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, seq_len),
        )

    def forward(self, cond_img):
        feat = self.encoder(cond_img)  # (B, 512)
        return self.head(feat)          # (B, 180)
