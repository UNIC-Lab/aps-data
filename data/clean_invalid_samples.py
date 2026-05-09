"""
扫描并删除包含 NaN/Inf 的异常样本（多进程加速版）
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import json


def check_single_sample(args):
    """
    检查单个样本（用于多进程）
    
    Returns:
        (idx, is_valid, error_msg, sample_name)
    """
    idx, file_path, sample_name, preprocessed_dir = args
    sample_path = Path(preprocessed_dir) / file_path
    
    if not sample_path.exists():
        return (idx, False, f"文件不存在: {sample_path}", sample_name)
    
    try:
        # 加载样本（使用 mmap_mode 加速，不完全加载到内存）
        data = np.load(sample_path, mmap_mode='r')
        
        # 快速检查：只检查是否有 NaN/Inf，不加载完整数据
        aps_norm = data['aps_norm']
        cond_img = data['cond_img']
        
        # 检查 NaN/Inf
        if np.any(np.isnan(aps_norm)) or np.any(np.isinf(aps_norm)):
            return (idx, False, "aps_norm 包含 NaN/Inf", sample_name)
        
        if np.any(np.isnan(cond_img)) or np.any(np.isinf(cond_img)):
            return (idx, False, "cond_img 包含 NaN/Inf", sample_name)
        
        return (idx, True, None, sample_name)
        
    except Exception as e:
        return (idx, False, f"加载错误: {str(e)}", sample_name)


def clean_invalid_samples(preprocessed_dir, num_workers=None, checkpoint_file=None):
    """
    扫描预处理后的数据，删除包含 NaN/Inf 的样本（多进程加速）
    
    Args:
        preprocessed_dir: 预处理数据目录
        num_workers: 工作进程数（默认为 CPU 核心数）
        checkpoint_file: 检查点文件路径（用于断点续传）
    """
    preprocessed_dir = Path(preprocessed_dir)
    metadata_file = preprocessed_dir / "metadata.csv"
    
    if not metadata_file.exists():
        print(f"错误: {metadata_file} 不存在")
        return
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = cpu_count()
    print(f"使用 {num_workers} 个进程并行检查")
    
    # 读取 metadata
    print("加载 metadata.csv...")
    df = pd.read_csv(metadata_file)
    print(f"总样本数: {len(df)}")
    
    # 检查是否有检查点
    checked_indices = set()
    invalid_samples = []
    
    if checkpoint_file and Path(checkpoint_file).exists():
        print(f"\n从检查点恢复: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            checked_indices = set(checkpoint_data['checked_indices'])
            invalid_samples = checkpoint_data['invalid_samples']
        print(f"已检查: {len(checked_indices)} 个样本")
        print(f"已发现异常: {len(invalid_samples)} 个样本")
    
    # 准备待检查的样本（只传递可序列化的数据）
    tasks = []
    for idx, row in df.iterrows():
        if idx not in checked_indices:
            # 只传递基本类型，避免传递 pandas Series
            tasks.append((idx, row['file_path'], row['sample_name'], str(preprocessed_dir)))
    
    if len(tasks) == 0:
        print("\n所有样本已检查完毕！")
    else:
        print(f"\n待检查样本: {len(tasks)} 个")
        print("开始并行扫描...")
        
        # 多进程并行检查
        results = []
        with Pool(processes=num_workers) as pool:
            # 使用 imap 并设置 chunksize
            pbar = tqdm(total=len(tasks), desc="检查样本", unit="样本")
            for result in pool.imap(check_single_sample, tasks, chunksize=100):
                results.append(result)
                pbar.update(1)
                
                # 实时显示发现的异常
                idx, is_valid, error_msg, sample_name = result
                if not is_valid:
                    pbar.write(f"发现异常 [{idx}]: {sample_name} - {error_msg}")
            pbar.close()
        
        # 收集结果
        for idx, is_valid, error_msg, sample_name in results:
            checked_indices.add(idx)
            if not is_valid:
                invalid_samples.append(idx)
        
        # 保存检查点
        if checkpoint_file:
            checkpoint_data = {
                'checked_indices': list(checked_indices),
                'invalid_samples': invalid_samples
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"\n检查点已保存: {checkpoint_file}")
    
    # 删除异常样本
    if len(invalid_samples) == 0:
        print("\n✓ 未发现异常样本，数据集正常！")
        return
    
    print(f"\n发现 {len(invalid_samples)} 个异常样本")
    print("删除异常样本...")
    
    deleted_count = 0
    for idx in invalid_samples:
        row = df.iloc[idx]
        sample_path = preprocessed_dir / row['file_path']
        
        # 删除文件
        if sample_path.exists():
            sample_path.unlink()
            deleted_count += 1
            if deleted_count <= 10:  # 只打印前 10 个
                print(f"  删除: {row['sample_name']}")
    
    if deleted_count > 10:
        print(f"  ... 共删除 {deleted_count} 个文件")
    
    # 更新 metadata.csv
    df_clean = df.drop(invalid_samples).reset_index(drop=True)
    df_clean.to_csv(metadata_file, index=False)
    
    print(f"\n清理完成！")
    print(f"  - 删除异常样本: {len(invalid_samples)} 个")
    print(f"  - 剩余样本: {len(df_clean)} 个")
    print(f"  - 已更新 metadata.csv")
    
    # 删除检查点文件
    if checkpoint_file and Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        print(f"  - 已删除检查点文件")


def main():
    parser = argparse.ArgumentParser(description="清理包含 NaN/Inf 的异常样本（多进程加速）")
    parser.add_argument("--preprocessed_dir", type=str, required=True,
                        help="预处理数据目录")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="工作进程数（默认为 CPU 核心数）")
    parser.add_argument("--checkpoint", type=str, default="clean_checkpoint.json",
                        help="检查点文件路径（用于断点续传）")
    
    args = parser.parse_args()
    clean_invalid_samples(args.preprocessed_dir, args.num_workers, args.checkpoint)


if __name__ == "__main__":
    main()
