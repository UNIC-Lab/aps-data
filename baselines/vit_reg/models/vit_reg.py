"""ViT Regression - Vision Transformer for direct APS regression"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTRegression(nn.Module):
    """
    ViT-based direct regression for APS prediction

    架构:
        - Backbone: ViT-B/16 (pretrained on ImageNet)
        - Head: MLP regression (768 → 512 → 180)

    输入: condition image (B, 3, 256, 256)
    输出: APS (B, 180)
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config['model']
        self.seq_len = model_cfg['seq_len']
        use_pretrained = model_cfg.get('use_pretrained', True)

        # ViT-B/16 backbone
        if use_pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)

        # ViT-B/16 输出 768 维
        vit_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()  # 去掉分类头

        # Regression head
        hidden_dim = model_cfg.get('hidden_dim', 512)
        self.head = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(model_cfg.get('dropout', 0.1)),
            nn.Linear(hidden_dim, self.seq_len),
        )

    def forward(self, cond_img):
        """
        Args:
            cond_img: (B, 3, H, W) condition image (will be resized to 224x224 by ViT)

        Returns:
            aps_pred: (B, 180)
        """
        # ViT-B/16 expects 224x224, resize
        if cond_img.shape[-1] != 224 or cond_img.shape[-2] != 224:
            cond_img = F.interpolate(cond_img, size=(224, 224), mode='bilinear', align_corners=False)
        feat = self.vit(cond_img)     # (B, 768)
        aps_pred = self.head(feat)    # (B, 180)
        return aps_pred
