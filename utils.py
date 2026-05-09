"""工具函数"""
import numpy as np
import torch
import yaml
import random
from pathlib import Path
from PIL import Image, ImageDraw


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rasterize_buildings(buildings_json, width, height):
    """
    将建筑物多边形光栅化为二值图
    
    Args:
        buildings_json: list of polygons, 每个 polygon 是 [[x1,y1], [x2,y2], ...]
        width, height: 输出图像尺寸
    
    Returns:
        numpy array (height, width) float32, 0=空地, 1=建筑
    """
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    for polygon in buildings_json:
        # polygon: [[x, y], ...] -> flatten to [x1, y1, x2, y2, ...]
        coords = []
        for point in polygon:
            coords.extend(point)
        if len(coords) >= 6:  # 至少3个点
            draw.polygon(coords, fill=255)
    
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def gaussian_heatmap(x, y, width, height, sigma=20.0):
    """
    生成高斯热力图，中心在 (x, y)，距离越近越亮
    
    Args:
        x, y: 中心坐标
        width, height: 图像尺寸
        sigma: 高斯标准差（控制衰减速度）
    
    Returns:
        numpy array (height, width) float32, 范围 [0, 1]
    """
    # 创建网格
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    # 计算距离
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    
    # 高斯函数
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    
    return heatmap.astype(np.float32)


def normalize_aps_db(aps_db, eps=1e-8):
    """
    直接在 dB 域做 z-score 归一化，不转换到线性域
    
    Args:
        aps_db: (180,) numpy array, 原始 dB 值
        eps: 防止除零
    
    Returns:
        aps_norm: (180,) float32, z-score 归一化后的 dB 值
    """
    aps_shifted = aps_db - aps_db.max()  # 平移使 max=0 dB，范围 (-∞, 0]
    mean = aps_shifted.mean()
    std = aps_shifted.std()
    aps_norm = (aps_shifted - mean) / (std + eps)
    return aps_norm.astype(np.float32)


def normalize_aps_std(aps_db, eps=1e-8):
    """Alternative normalization: divide by standard deviation (z-score)."""
    aps_shifted = aps_db - aps_db.max()
    aps_linear = 10 ** (aps_shifted / 10.0)  # Range (0, 1]

    mean = aps_linear.mean()
    std = aps_linear.std()
    aps_norm = (aps_linear - mean) / (std + eps)
    return aps_norm.astype(np.float32)


def normalize_aps_minmax(aps_db):
    """Legacy normalization: dB -> linear -> [-1, 1]."""
    aps_shifted = aps_db - aps_db.max()
    aps_linear = 10 ** (aps_shifted / 10.0)
    aps_norm = aps_linear * 2.0 - 1.0
    return aps_norm.astype(np.float32)


def normalize_aps(aps_db):
    """Backward-compatible alias. Default now uses std normalization."""
    return normalize_aps_std(aps_db)


def denormalize_aps(aps_norm):
    """
    反归一化（仅适用于 minmax 归一化结果，主要用于可视化）
    
    Args:
        aps_norm: (180,) numpy array, [-1, 1]
    
    Returns:
        aps_db_relative: (180,) numpy array, 相对 dB 值
    """
    # [-1, 1] -> [0, 1]
    aps_linear = (aps_norm + 1.0) / 2.0
    aps_linear = np.maximum(aps_linear, 1e-10)  # 只防止 log10(0)，允许超过 1.0
    
    # 转 dB（相对值）
    aps_db_relative = 10 * np.log10(aps_linear)
    
    return aps_db_relative


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
