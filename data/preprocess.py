"""
数据预处理脚本
一次性运行，生成预处理后的数据
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd

from utils import (
    rasterize_buildings,
    gaussian_heatmap,
    normalize_aps_std,
    normalize_aps_minmax,
    normalize_aps_db,
)


def parse_filename(filename):
    """
    从文件名解析信息
    例如: aps_0_0_170_0_88.npy -> map_id=0, tx_x=0, tx_y=170, rx_x=0, rx_y=88
    """
    parts = filename.replace('.npy', '').split('_')
    if len(parts) >= 6 and parts[0] == 'aps':
        return {
            'map_id': int(parts[1]),
            'tx_x': int(parts[2]),
            'tx_y': int(parts[3]),
            'rx_x': int(parts[4]),
            'rx_y': int(parts[5])
        }
    return None


def preprocess_data(
    aps_root_dir,
    buildings_dir,
    output_dir,
    map_ids=None,
    img_size=256,
    sigma=20.0,
    aps_norm="std",
):
    """
    预处理数据
    
    Args:
        aps_root_dir: APS 数据根目录，包含 map_0, map_1, ... 子目录
        buildings_dir: 建筑物 JSON 文件目录
        output_dir: 输出目录
        map_ids: 要处理的 map id 列表，None 表示处理所有
        img_size: condition 图像尺寸
        sigma: 高斯热力图的标准差
        aps_norm: APS 归一化方法，'std' 或 'minmax'
    """
    output_dir = Path(output_dir)
    env_maps_dir = output_dir / "env_maps"
    samples_dir = output_dir / "samples"
    
    env_maps_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有 APS 文件
    aps_root = Path(aps_root_dir)
    all_aps_files = []
    
    # 遍历 map_* 目录
    for map_dir in sorted(aps_root.glob("map_*")):
        map_id = int(map_dir.name.split('_')[1])
        
        if map_ids is not None and map_id not in map_ids:
            continue
        
        # 查找 APS 文件（在 *_adps/aps/ 子目录下）
        aps_dir = map_dir / f"{map_id}_adps" / "aps"
        if not aps_dir.exists():
            print(f"警告: {aps_dir} 不存在，跳过")
            continue
        
        for aps_file in aps_dir.glob("aps_*.npy"):
            all_aps_files.append((map_id, aps_file))
    
    print(f"找到 {len(all_aps_files)} 个 APS 文件")
    print(f"APS 归一化方法: {aps_norm}")
    
    # 预处理环境图（每个 map 只处理一次）
    processed_maps = set()
    
    # 尝试加载已有的 metadata.csv，实现断点续传
    metadata_file = output_dir / "metadata.csv"
    existing_samples = set()
    metadata = []
    
    if metadata_file.exists():
        print(f"发现已有 metadata.csv，加载已处理的样本...")
        existing_df = pd.read_csv(metadata_file)
        metadata = existing_df.to_dict('records')
        existing_samples = set(existing_df['sample_name'].tolist())
        print(f"已处理 {len(existing_samples)} 个样本，将跳过这些样本")
    
    # 统计跳过和新处理的样本数
    skipped_count = 0
    new_count = 0
    invalid_count = 0
    invalid_log = []  # 记录异常样本
    
    for map_id, aps_file in tqdm(all_aps_files, desc="预处理数据"):
        # 检查是否已处理过该样本
        sample_name = aps_file.stem  # 去掉 .npy
        if sample_name in existing_samples:
            skipped_count += 1
            continue
        
        # 1. 处理环境图
        if map_id not in processed_maps:
            buildings_json_path = Path(buildings_dir) / f"{map_id}.json"
            if not buildings_json_path.exists():
                print(f"警告: {buildings_json_path} 不存在，跳过 map {map_id}")
                continue
            
            with open(buildings_json_path, 'r') as f:
                buildings = json.load(f)
            
            env_map = rasterize_buildings(buildings, img_size, img_size)
            np.save(env_maps_dir / f"{map_id}.npy", env_map)
            processed_maps.add(map_id)
        
        # 2. 加载 APS 数据
        try:
            aps_data = np.load(aps_file)  # (180, 2)
            aps_db = aps_data[:, 0]  # 取功率列
        except Exception as e:
            print(f"错误: 无法加载 {aps_file}: {e}")
            continue
        
        # 3. 检查原始 APS 数据是否有效
        if np.any(np.isnan(aps_db)) or np.any(np.isinf(aps_db)):
            invalid_count += 1
            invalid_log.append({
                'sample_name': sample_name,
                'map_id': map_id,
                'reason': 'aps_db 包含 NaN/Inf'
            })
            continue
        
        # 4. 归一化 APS
        if aps_norm == "std":
            aps_norm_arr = normalize_aps_std(aps_db)
        elif aps_norm == "minmax":
            aps_norm_arr = normalize_aps_minmax(aps_db)
        elif aps_norm == "db":
            aps_norm_arr = normalize_aps_db(aps_db)
        else:
            raise ValueError(f"不支持的归一化方法: {aps_norm}")
        
        # 5. 检查归一化后的 APS 是否有效
        if np.any(np.isnan(aps_norm_arr)) or np.any(np.isinf(aps_norm_arr)):
            invalid_count += 1
            invalid_log.append({
                'sample_name': sample_name,
                'map_id': map_id,
                'reason': 'aps_norm 包含 NaN/Inf（归一化后）'
            })
            continue
        
        # 6. 解析文件名获取坐标
        info = parse_filename(aps_file.name)
        if info is None:
            print(f"警告: 无法解析文件名 {aps_file.name}")
            continue
        
        tx_x, tx_y = info['tx_x'], info['tx_y']
        rx_x, rx_y = info['rx_x'], info['rx_y']
        
        # 7. 生成 Tx/Rx 热力图
        tx_heatmap = gaussian_heatmap(tx_x, tx_y, img_size, img_size, sigma)
        rx_heatmap = gaussian_heatmap(rx_x, rx_y, img_size, img_size, sigma)
        
        # 8. 加载环境图
        env_map = np.load(env_maps_dir / f"{map_id}.npy")
        
        # 9. 拼接 condition 图 (3, H, W)
        cond_img = np.stack([env_map, tx_heatmap, rx_heatmap], axis=0).astype(np.float32)
        
        # 10. 检查 condition 图是否有效
        if np.any(np.isnan(cond_img)) or np.any(np.isinf(cond_img)):
            invalid_count += 1
            invalid_log.append({
                'sample_name': sample_name,
                'map_id': map_id,
                'reason': 'cond_img 包含 NaN/Inf'
            })
            continue
        
        # 11. 保存样本
        sample_path = samples_dir / f"{sample_name}.npz"
        np.savez_compressed(sample_path, aps_norm=aps_norm_arr, cond_img=cond_img)
        
        # 12. 记录元数据
        metadata.append({
            'sample_name': sample_name,
            'map_id': map_id,
            'tx_x': tx_x,
            'tx_y': tx_y,
            'rx_x': rx_x,
            'rx_y': rx_y,
            'aps_norm': aps_norm,
            'file_path': str(sample_path.relative_to(output_dir))
        })
        new_count += 1
    
    # 保存元数据
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    # 保存异常样本日志
    if invalid_log:
        invalid_df = pd.DataFrame(invalid_log)
        invalid_log_file = output_dir / "invalid_samples.csv"
        invalid_df.to_csv(invalid_log_file, index=False)
        print(f"\n⚠️  发现 {invalid_count} 个异常样本（包含 NaN/Inf），已跳过")
        print(f"  - 异常样本日志: {invalid_log_file}")
        if invalid_count <= 10:
            print(f"  - 异常样本列表:")
            for item in invalid_log:
                print(f"    • {item['sample_name']}: {item['reason']}")
        else:
            print(f"  - 前 10 个异常样本:")
            for item in invalid_log[:10]:
                print(f"    • {item['sample_name']}: {item['reason']}")
    
    print(f"\n✓ 预处理完成！")
    print(f"  - 处理了 {len(processed_maps)} 个 map")
    print(f"  - 跳过已存在的样本: {skipped_count} 个")
    print(f"  - 跳过异常样本: {invalid_count} 个")
    print(f"  - 新处理的样本: {new_count} 个")
    print(f"  - 有效样本总数: {len(metadata)} 个")
    print(f"  - 输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="预处理 APS 数据")
    parser.add_argument("--aps_root", type=str, default=".",
                        help="APS 数据根目录（包含 map_0, map_1, ... 子目录）")
    parser.add_argument("--buildings_dir", type=str, default="buildings_complete",
                        help="建筑物 JSON 文件目录")
    parser.add_argument("--output_dir", type=str, default="preprocessed",
                        help="输出目录")
    parser.add_argument("--map_ids", type=str, default=None,
                        help="要处理的 map id，逗号分隔，例如 '0,1,2'，默认处理所有")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Condition 图像尺寸")
    parser.add_argument("--sigma", type=float, default=20.0,
                        help="高斯热力图标准差")
    parser.add_argument(
        "--aps_norm",
        type=str,
        default="std",
        choices=["std", "minmax", "db"],
        help="APS 归一化方法：std(标准差归一化) / minmax(映射到[-1,1]) / db(直接在dB域做z-score)",
    )
    
    args = parser.parse_args()
    
    map_ids = None
    if args.map_ids:
        map_ids = [int(x.strip()) for x in args.map_ids.split(',')]
    
    preprocess_data(
        aps_root_dir=args.aps_root,
        buildings_dir=args.buildings_dir,
        output_dir=args.output_dir,
        map_ids=map_ids,
        img_size=args.img_size,
        sigma=args.sigma,
        aps_norm=args.aps_norm,
    )


if __name__ == "__main__":
    main()
