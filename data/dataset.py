"""Dataset for APS Flow Matching"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob


class APSDataset(Dataset):
    """
    APS Dataset
    
    加载预处理好的数据（.npz 文件）
    每个样本包含：
        - aps_norm: (180,) float32, 归一化后的 APS 功率谱
        - cond_img: (3, 256, 256) float32, 三通道 condition 图
    """
    
    def __init__(self, preprocessed_dir, split='train', train_maps=None, val_maps=None):
        """
        Args:
            preprocessed_dir: 预处理数据目录
            split: 'train' or 'val'
            train_maps: 训练集地图编号列表，如 [0,1,2,...,45]
            val_maps: 验证集地图编号列表，如 [46,47,48,49,50]
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split
        
        # 默认划分：train=0-45, val=46-50
        if train_maps is None:
            train_maps = list(range(0, 46))
        if val_maps is None:
            val_maps = list(range(46, 51))
        
        # 收集所有样本文件
        samples_dir = self.preprocessed_dir / "samples"
        all_files = sorted(glob(str(samples_dir / "*.npz")))
        
        if len(all_files) == 0:
            raise ValueError(f"在 {samples_dir} 中未找到任何 .npz 文件")
        
        # 根据地图编号过滤
        target_maps = train_maps if split == 'train' else val_maps
        self.files = []
        
        for file_path in all_files:
            # 从文件名提取 map_id
            # 文件名格式: aps_{map_id}_{tx_id}_{rx_id}_{tx_x}_{tx_y}.npz
            filename = Path(file_path).stem
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    map_id = int(parts[1])
                    if map_id in target_maps:
                        self.files.append(file_path)
                except ValueError:
                    continue
        
        print(f"[{split}] 地图编号: {sorted(target_maps)}")
        print(f"[{split}] 加载了 {len(self.files)} 个样本")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # 加载 .npz 文件
        data = np.load(self.files[idx])
        
        aps_norm = data['aps_norm']      # (180,)
        cond_img = data['cond_img']      # (3, 256, 256)
        
        # 转为 torch tensor
        aps_norm = torch.from_numpy(aps_norm).float()
        cond_img = torch.from_numpy(cond_img).float()
        
        return {
            'aps': aps_norm,
            'cond': cond_img
        }


def get_dataloaders(config, world_size=1, rank=0):
    """
    创建 DataLoader
    
    Args:
        config: 配置字典
        world_size: DDP 总进程数
        rank: 当前进程 rank
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # 硬编码地图划分：train=0-45, val=46-50
    train_maps = list(range(0, 46))
    val_maps = list(range(46, 51))
    
    # 创建 Dataset
    train_dataset = APSDataset(
        preprocessed_dir=config['data']['preprocessed_dir'],
        split='train',
        train_maps=train_maps,
        val_maps=val_maps
    )
    
    val_dataset = APSDataset(
        preprocessed_dir=config['data']['preprocessed_dir'],
        split='val',
        train_maps=train_maps,
        val_maps=val_maps
    )
    
    # 创建 Sampler（多卡时使用 DistributedSampler）
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        # 验证/测试由 rank0 执行时，保持完整验证集，不做分片
        val_sampler = None
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=False
    )
    
    return train_loader, val_loader
