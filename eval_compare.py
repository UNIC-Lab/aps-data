"""统一对比评估脚本 - 支持全量/采样两种模式，均用 DataLoader 批量推理"""
import sys
import csv
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from utils import set_seed
from data.dataset import APSDataset


# ============================================================
# 工具函数
# ============================================================

def parse_filename(filepath):
    stem = Path(filepath).stem
    parts = stem.split('_')
    if len(parts) >= 6:
        return {
            'map_id': int(parts[1]),
            'tx_id': int(parts[2]),
            'rx_id': int(parts[3]),
            'tx_x': parts[4],
            'tx_y': parts[5],
            'filename': stem,
        }
    return {'map_id': -1, 'tx_id': -1, 'rx_id': -1, 'tx_x': '', 'tx_y': '', 'filename': stem}


def select_balanced_indices(dataset, num_samples, val_maps):
    """按 map 均匀选取样本索引"""
    from collections import defaultdict
    map_groups = defaultdict(list)
    for idx, filepath in enumerate(dataset.files):
        meta = parse_filename(filepath)
        map_groups[meta['map_id']].append(idx)

    num_per_map = num_samples // len(val_maps)
    selected = []

    print(f"\n=== 均衡采样 ===")
    print(f"  总样本数: {num_samples}, 验证 maps: {val_maps}, 每 map: {num_per_map}")
    for map_id in sorted(val_maps):
        items = map_groups.get(map_id, [])
        np.random.shuffle(items)
        chosen = items[:num_per_map]
        selected.extend(chosen)
        print(f"  Map {map_id}: 共 {len(items)} 个，选 {len(chosen)} 个")

    print(f"  实际总选: {len(selected)} 个\n")
    return selected


def compute_global_data_range(dataset, indices=None):
    """计算全局 data_range (GT 的 max - min)，扫描全部指定样本"""
    if indices is None:
        indices = list(range(len(dataset)))

    global_max = -np.inf
    global_min = np.inf
    for idx in tqdm(indices, desc="计算 data_range", leave=False):
        gt = dataset[idx]['aps'].numpy()
        global_max = max(global_max, gt.max())
        global_min = min(global_min, gt.min())

    data_range = max(global_max - global_min, 1e-6)
    print(f"全局 data_range: {data_range:.4f} (min={global_min:.4f}, max={global_max:.4f})")
    return data_range


def compute_peak_error(gt, pred, min_energy_ratio=0.10, min_prominence=0.05):
    """
    峰位置误差 (Peak Localization Error, PLE)

    动态确定有效峰：每个峰的高度占所有峰高度之和的比例 >= min_energy_ratio (默认 10%)。
    例如：峰能量占比 [45%, 40%, 5%] → 选前两个，丢掉第三个。

    - GT 决定有效峰数，pred 多余的假峰不参与惩罚
    - 若某 GT 峰在 pred 中无对应峰，计为 180°
    - 单位：度（每个 bin = 1°，共 180°）
    """
    from scipy.signal import find_peaks

    gt_norm = gt / (gt.max() + 1e-8)
    pred_norm = pred / (pred.max() + 1e-8)

    gt_peaks, _ = find_peaks(gt_norm, prominence=min_prominence)
    pred_peaks, _ = find_peaks(pred_norm, prominence=min_prominence)

    if len(gt_peaks) == 0:
        return 0.0  # GT 无峰（平坦/扩散信号），跳过

    # 每个峰的能量占比，过滤掉占比低于阈值的次要峰
    gt_heights = gt_norm[gt_peaks]
    total_height = gt_heights.sum()
    energy_ratio = gt_heights / (total_height + 1e-8)
    selected_gt = gt_peaks[energy_ratio >= min_energy_ratio]

    if len(selected_gt) == 0:
        selected_gt = gt_peaks[np.argmax(gt_heights):np.argmax(gt_heights)+1]  # 至少保留最高峰

    if len(pred_peaks) == 0:
        return 180.0

    # 对每个有效 GT 峰，在 pred 中找最近的峰
    errors = [float(np.abs(pred_peaks - gp).min()) for gp in selected_gt]
    return float(np.mean(errors))


def extract_dominant_peaks(aps, min_energy_ratio=0.10, min_prominence=0.05):
    """提取 dominant peaks（用于 Hit Rate 和 Recall）

    返回：peaks (ndarray of indices), heights (ndarray)
    """
    from scipy.signal import find_peaks

    aps_norm = aps / (aps.max() + 1e-8)
    peaks, _ = find_peaks(aps_norm, prominence=min_prominence)

    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([])

    heights = aps_norm[peaks]
    total_height = heights.sum()
    energy_ratio = heights / (total_height + 1e-8)
    selected = peaks[energy_ratio >= min_energy_ratio]

    if len(selected) == 0:
        # 至少保留最高峰
        max_idx = np.argmax(heights)
        selected = np.array([peaks[max_idx]])

    return selected, aps_norm[selected]


def compute_top1_hit_rate(gt, pred, delta=2.0, min_energy_ratio=0.10, min_prominence=0.05):
    """Top-1 Dominant Peak Hit Rate@δ

    只看最强主峰是否命中，返回 0/1
    """
    gt_peaks, gt_heights = extract_dominant_peaks(gt, min_energy_ratio, min_prominence)
    pred_peaks, pred_heights = extract_dominant_peaks(pred, min_energy_ratio, min_prominence)

    if len(gt_peaks) == 0 or len(pred_peaks) == 0:
        return 0.0

    # GT 和 pred 中最高的峰
    gt_top1 = gt_peaks[np.argmax(gt_heights)]
    pred_top1 = pred_peaks[np.argmax(pred_heights)]

    # 角度距离（bin 差 = 角度差，因为每 bin 是 1°，共 180 bin）
    ang_dist = float(np.abs(gt_top1 - pred_top1))
    return float(ang_dist <= delta)


def compute_dominant_peak_recall(gt, pred, delta=2.0, min_energy_ratio=0.10, min_prominence=0.05):
    """Dominant Peak Recall@δ

    GT 中有多少 dominant peak 在 pred 中被找回，返回 [0, 1]
    """
    gt_peaks, _ = extract_dominant_peaks(gt, min_energy_ratio, min_prominence)
    pred_peaks, _ = extract_dominant_peaks(pred, min_energy_ratio, min_prominence)

    if len(gt_peaks) == 0:
        return 1.0  # GT 没有峰，默认 recall 100%

    if len(pred_peaks) == 0:
        return 0.0  # GT 有峰但 pred 没有，recall 0%

    # 对每个 GT peak，检查是否存在 pred peak 在容差范围内
    recalled = 0
    for gp in gt_peaks:
        min_dist = np.abs(pred_peaks - gp).min()
        if min_dist <= delta:
            recalled += 1

    return float(recalled / len(gt_peaks))


# ============================================================
# 模型加载
# ============================================================

def load_model(model_type, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    if model_type == 'ms_areg':
        from baselines.ms_areg.models import MSAReg
        config['model']['use_pretrained'] = False
        model = MSAReg(config).to(device)
        model.predictor.load_state_dict(ckpt['p_state_dict'])
        return model, config

    elif model_type == 'adv_mlp':
        from baselines.adv_mlp.models import AdvMLP
        model = AdvMLP(config).to(device)
        model.generator.load_state_dict(ckpt['g_state_dict'])
        return model, config

    elif model_type == 'radiounet':
        from baselines.radiounet.models import RadioUNet
        model = RadioUNet(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, config

    elif model_type == 'vit_reg':
        from baselines.vit_reg.models import ViTRegression
        config['model']['use_pretrained'] = False
        model = ViTRegression(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, config

    elif model_type == 'resnet_mlp':
        from baselines.resnet_mlp.models import ResNetMLP
        config['model']['use_pretrained'] = False
        model = ResNetMLP(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, config

    elif model_type == 'resnet_conv1d':
        from baselines.resnet_conv1d.models import ResNetConv1D
        config['model']['use_pretrained'] = False
        model = ResNetConv1D(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, config

    elif model_type == 'ms_mlp':
        from baselines.ms_mlp.models import MSMLP
        config['model']['use_pretrained'] = False
        model = MSMLP(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, config

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# 批量推理
# ============================================================

@torch.no_grad()
def infer_batch(model, model_type, cond, device):
    """批量推理，返回 (B, 180) numpy"""
    cond = cond.to(device)
    B = cond.shape[0]

    if model_type == 'ms_areg':
        aps_pred = model.predictor(cond)
    elif model_type == 'adv_mlp':
        z = torch.zeros(B, model.z_dim, device=device)
        aps_pred = model.generator(cond, z)
    elif model_type in ('radiounet', 'vit_reg', 'resnet_mlp', 'resnet_conv1d', 'ms_mlp'):
        aps_pred = model(cond)

    return aps_pred.cpu().numpy()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='统一对比评估（支持全量/采样模式）')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='模型列表, 格式: type:checkpoint_path')
    parser.add_argument('--data_dir', type=str, default='/home/DataDisk/jxhuang/aps_preprocessed')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'sample'],
                       help='full: 全量验证集 (~25万); sample: 均衡采样')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='采样模式下的总样本数 (按 map 均分)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='/home/DataDisk/jxhuang/results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载验证集
    val_maps = list(range(46, 51))
    val_dataset = APSDataset(
        preprocessed_dir=args.data_dir,
        split='val',
        val_maps=val_maps,
    )

    # 根据模式确定评估数据集和元数据
    if args.mode == 'sample':
        selected_indices = select_balanced_indices(val_dataset, args.num_samples, val_maps)
        eval_dataset = Subset(val_dataset, selected_indices)
        # 元数据按 selected_indices 顺序
        all_meta = [parse_filename(val_dataset.files[i]) for i in selected_indices]
        print(f"采样模式: {len(eval_dataset)} 个样本")
    else:
        eval_dataset = val_dataset
        all_meta = [parse_filename(f) for f in val_dataset.files]
        print(f"全量模式: {len(eval_dataset)} 个样本")

    # 计算全局 data_range（全量扫描，保证覆盖所有 val map）
    if args.mode == 'sample':
        global_data_range = compute_global_data_range(val_dataset, selected_indices)
    else:
        global_data_range = compute_global_data_range(val_dataset)

    # DataLoader
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ============================================================
    # 逐模型评估
    # ============================================================
    from multiprocessing import Pool

    all_summary = {}

    for model_spec in args.models:
        model_type, ckpt_path = model_spec.split(':', 1)
        print(f"\n{'='*60}")
        print(f">>> 评估 {model_type} ({ckpt_path})")
        print(f"{'='*60}")

        model, _ = load_model(model_type, ckpt_path, device)
        model.eval()

        # === Step 1: 推理，收集全部 gt / pred ===
        all_gt = []
        all_pred = []
        total_infer_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Infer {model_type}"):
                cond = batch['cond']
                gt_batch = batch['aps']  # (B, 180) tensor
                B = cond.shape[0]

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred_batch = infer_batch(model, model_type, cond, device)  # (B, 180) numpy
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                all_gt.append(gt_batch)
                all_pred.append(torch.from_numpy(pred_batch))
                total_infer_time += elapsed
                total_samples += B

        del model
        torch.cuda.empty_cache()

        gt_all = torch.cat(all_gt, dim=0).float()    # (N, 180)
        pred_all = torch.cat(all_pred, dim=0).float() # (N, 180)
        infer_time_per_sample = total_infer_time / total_samples
        infer_time_total = total_infer_time

        # === Step 2: 批量 GPU 指标 ===
        print(f"  计算 GPU 指标...")
        gt_gpu = gt_all.to(device)
        pred_gpu = pred_all.to(device)

        diff = gt_gpu - pred_gpu
        mae_vec   = diff.abs().mean(dim=1).cpu().numpy()           # (N,)
        mse_vec   = (diff ** 2).mean(dim=1).cpu().numpy()          # (N,)
        rmse_vec  = np.sqrt(mse_vec)

        gt_pow      = (gt_gpu ** 2).mean(dim=1).clamp(min=1e-12).cpu().numpy()
        nmse_vec    = mse_vec / gt_pow   # 复用已算好的 mse_vec
        nmse_db_vec = 10 * np.log10(np.maximum(nmse_vec, 1e-12))

        cos_vec   = F.cosine_similarity(gt_gpu, pred_gpu, dim=1).clamp(-1, 1).cpu().numpy()
        angle_vec = np.degrees(np.arccos(cos_vec))

        psnr_vec  = 20 * np.log10(global_data_range) - 10 * np.log10(np.maximum(mse_vec, 1e-12))

        del gt_gpu, pred_gpu
        torch.cuda.empty_cache()

        # === Step 3: Peak-based metrics (多进程) ===
        print(f"  计算 PLE, Top-1 Hit Rate, Dominant Peak Recall...")
        gt_np   = gt_all.numpy()
        pred_np = pred_all.numpy()
        pairs   = [(gt_np[i], pred_np[i]) for i in range(len(gt_np))]

        with Pool(processes=args.num_workers) as pool:
            ple_vec = np.array(pool.starmap(compute_peak_error, pairs, chunksize=500))
            hit_rate_2_vec = np.array(pool.starmap(compute_top1_hit_rate,
                                                   [(gt_np[i], pred_np[i], 2.0) for i in range(len(gt_np))],
                                                   chunksize=500))
            hit_rate_4_vec = np.array(pool.starmap(compute_top1_hit_rate,
                                                   [(gt_np[i], pred_np[i], 4.0) for i in range(len(gt_np))],
                                                   chunksize=500))
            recall_2_vec = np.array(pool.starmap(compute_dominant_peak_recall,
                                                 [(gt_np[i], pred_np[i], 2.0) for i in range(len(gt_np))],
                                                 chunksize=500))
            recall_4_vec = np.array(pool.starmap(compute_dominant_peak_recall,
                                                 [(gt_np[i], pred_np[i], 4.0) for i in range(len(gt_np))],
                                                 chunksize=500))

        # === Step 4: 整理 per-sample results ===
        per_sample_results = []
        for i in range(len(gt_all)):
            per_sample_results.append({
                **all_meta[i],
                'MAE': float(mae_vec[i]), 'MSE': float(mse_vec[i]), 'RMSE': float(rmse_vec[i]),
                'PSNR': float(psnr_vec[i]), 'CosSim': float(cos_vec[i]),
                'NMSE_dB': float(nmse_db_vec[i]), 'SpectralAngle': float(angle_vec[i]),
                'PLE': float(ple_vec[i]),
                'Top1_Hit_Rate@2deg': float(hit_rate_2_vec[i]),
                'Top1_Hit_Rate@4deg': float(hit_rate_4_vec[i]),
                'Dominant_Peak_Recall@2deg': float(recall_2_vec[i]),
                'Dominant_Peak_Recall@4deg': float(recall_4_vec[i]),
                'infer_time': infer_time_per_sample,
            })

        # --- 逐样本 CSV ---
        detail_path = output_dir / f'{model_type}_detail.csv'
        with open(detail_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'map_id', 'tx_id', 'rx_id', 'tx_x', 'tx_y',
                           'MAE', 'MSE', 'RMSE', 'PSNR', 'CosSim',
                           'NMSE_dB', 'SpectralAngle', 'PLE',
                           'Top1_Hit_Rate@2deg', 'Top1_Hit_Rate@4deg',
                           'Dominant_Peak_Recall@2deg', 'Dominant_Peak_Recall@4deg',
                           'infer_time(s)'])
            for r in per_sample_results:
                writer.writerow([
                    r['filename'], r['map_id'], r['tx_id'], r['rx_id'], r['tx_x'], r['tx_y'],
                    f"{r['MAE']:.6f}", f"{r['MSE']:.6f}", f"{r['RMSE']:.6f}",
                    f"{r['PSNR']:.2f}", f"{r['CosSim']:.4f}",
                    f"{r['NMSE_dB']:.2f}", f"{r['SpectralAngle']:.4f}", f"{r['PLE']:.4f}",
                    f"{r['Top1_Hit_Rate@2deg']:.1f}", f"{r['Top1_Hit_Rate@4deg']:.1f}",
                    f"{r['Dominant_Peak_Recall@2deg']:.4f}", f"{r['Dominant_Peak_Recall@4deg']:.4f}",
                    f"{r['infer_time']:.6f}",
                ])
        print(f"  逐样本 CSV: {detail_path}")

        # --- 汇总统计 ---
        summary = {
            'model': model_type, 'checkpoint': ckpt_path,
            'num_samples': len(per_sample_results),
            'MAE_mean': float(mae_vec.mean()),       'MAE_std': float(mae_vec.std()),
            'MSE_mean': float(mse_vec.mean()),       'MSE_std': float(mse_vec.std()),
            'RMSE_mean': float(rmse_vec.mean()),     'RMSE_std': float(rmse_vec.std()),
            'PSNR_mean': float(psnr_vec.mean()),     'PSNR_std': float(psnr_vec.std()),
            'CosSim_mean': float(cos_vec.mean()),    'CosSim_std': float(cos_vec.std()),
            'NMSE_dB_mean': float(nmse_db_vec.mean()), 'NMSE_dB_std': float(nmse_db_vec.std()),
            'SpectralAngle_mean': float(angle_vec.mean()), 'SpectralAngle_std': float(angle_vec.std()),
            'PLE_mean': float(ple_vec.mean()),       'PLE_std': float(ple_vec.std()),
            'Top1_Hit_Rate@2deg_mean': float(hit_rate_2_vec.mean()),
            'Top1_Hit_Rate@4deg_mean': float(hit_rate_4_vec.mean()),
            'Dominant_Peak_Recall@2deg_mean': float(recall_2_vec.mean()),
            'Dominant_Peak_Recall@4deg_mean': float(recall_4_vec.mean()),
            'infer_time_per_sample': infer_time_per_sample,
            'infer_time_total': infer_time_total,
        }

        map_ids = np.array([m['map_id'] for m in all_meta])
        for map_id in sorted(val_maps):
            mask = map_ids == map_id
            if mask.any():
                summary[f'map{map_id}_MAE']   = float(mae_vec[mask].mean())
                summary[f'map{map_id}_PSNR']  = float(psnr_vec[mask].mean())
                summary[f'map{map_id}_count'] = int(mask.sum())

        all_summary[model_type] = summary

        print(f"  样本数:           {summary['num_samples']}")
        print(f"  MAE:              {summary['MAE_mean']:.6f} ± {summary['MAE_std']:.6f}")
        print(f"  PSNR:             {summary['PSNR_mean']:.2f} ± {summary['PSNR_std']:.2f} dB")
        print(f"  CosSim:           {summary['CosSim_mean']:.4f} ± {summary['CosSim_std']:.4f}")
        print(f"  NMSE:             {summary['NMSE_dB_mean']:.2f} ± {summary['NMSE_dB_std']:.2f} dB")
        print(f"  SpectralAngle:    {summary['SpectralAngle_mean']:.4f} ± {summary['SpectralAngle_std']:.4f} °")
        print(f"  PLE:              {summary['PLE_mean']:.4f} ± {summary['PLE_std']:.4f} °")
        print(f"  Top-1 Hit@2°:     {summary['Top1_Hit_Rate@2deg_mean']:.4f}")
        print(f"  Top-1 Hit@4°:     {summary['Top1_Hit_Rate@4deg_mean']:.4f}")
        print(f"  Peak Recall@2°:   {summary['Dominant_Peak_Recall@2deg_mean']:.4f}")
        print(f"  Peak Recall@4°:   {summary['Dominant_Peak_Recall@4deg_mean']:.4f}")
        print(f"  推理速度:         {infer_time_per_sample*1000:.3f} ms/sample")

    # ============================================================
    # 总 summary CSV
    # ============================================================
    summary_path = output_dir / 'summary.csv'
    if all_summary:
        all_keys = list(list(all_summary.values())[0].keys())
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for name, s in all_summary.items():
                formatted = {}
                for k, v in s.items():
                    if isinstance(v, float):
                        formatted[k] = f"{v:.6f}" if abs(v) < 100 else f"{v:.2f}"
                    else:
                        formatted[k] = v
                writer.writerow(formatted)
        print(f"\n总 summary CSV: {summary_path}")

    # ============================================================
    # 打印对比表
    # ============================================================
    print("\n" + "=" * 180)
    print(f"{'Model':<15} {'MAE':>10} {'PSNR(dB)':>10} {'CosSim':>10} {'NMSE(dB)':>10} {'Angle(°)':>10} {'PLE(°)':>10} "
          f"{'Hit@2°':>10} {'Hit@4°':>10} {'Recall@2°':>10} {'Recall@4°':>10} {'ms/sample':>10}")
    print("-" * 180)
    for name, s in all_summary.items():
        print(f"{name:<15} {s['MAE_mean']:>10.6f} {s['PSNR_mean']:>10.2f} {s['CosSim_mean']:>10.4f} "
              f"{s['NMSE_dB_mean']:>10.2f} {s['SpectralAngle_mean']:>10.4f} {s['PLE_mean']:>10.4f} "
              f"{s['Top1_Hit_Rate@2deg_mean']:>10.4f} {s['Top1_Hit_Rate@4deg_mean']:>10.4f} "
              f"{s['Dominant_Peak_Recall@2deg_mean']:>10.4f} {s['Dominant_Peak_Recall@4deg_mean']:>10.4f} "
              f"{s['infer_time_per_sample']*1000:>10.3f}")
    print("=" * 180)


if __name__ == '__main__':
    main()
