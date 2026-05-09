"""Sampling script for RadioUNet Baseline"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import load_config, set_seed
from data.dataset import get_dataloaders
from baselines.radiounet.models import RadioUNet


def visualize_aps(aps_gt, aps_pred, save_path):
    angles = np.arange(len(aps_gt))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(angles, aps_gt, label='Ground Truth', linewidth=2)
    plt.plot(angles, aps_pred, label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Angle Index'); plt.ylabel('Normalized Power')
    plt.title('APS Comparison'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    error = np.abs(aps_gt - aps_pred)
    plt.plot(angles, error, color='red', linewidth=2)
    plt.xlabel('Angle Index'); plt.ylabel('Absolute Error')
    plt.title(f'Error (MAE: {error.mean():.4f})'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()


def compute_metrics(results):
    maes, mses, psnrs, cosine_sims = [], [], [], []
    for res in results:
        gt, pred = res['gt'], res['pred']
        mae = np.abs(gt - pred).mean()
        mse = ((gt - pred) ** 2).mean()
        data_range = max(float(gt.max() - gt.min()), 1e-6)
        psnr = 20 * np.log10(data_range) - 10 * np.log10(max(mse, 1e-12))
        cosine_sim = np.dot(gt, pred) / (np.linalg.norm(gt) * np.linalg.norm(pred) + 1e-8)
        maes.append(mae); mses.append(mse); psnrs.append(psnr); cosine_sims.append(cosine_sim)
    return {
        'MAE': np.mean(maes), 'MAE_std': np.std(maes),
        'MSE': np.mean(mses), 'MSE_std': np.std(mses),
        'RMSE': np.sqrt(np.mean(mses)),
        'PSNR': np.mean(psnrs), 'PSNR_std': np.std(psnrs),
        'Cosine_Similarity': np.mean(cosine_sims), 'Cosine_Similarity_std': np.std(cosine_sims),
    }


def main():
    parser = argparse.ArgumentParser(description='RadioUNet Sampling')
    parser.add_argument('--config', type=str, default='baselines/radiounet/config_radiounet.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='baselines/radiounet/samples')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_vis', type=int, default=10)
    parser.add_argument('--save_json', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = RadioUNet(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"已加载 checkpoint: {args.checkpoint}, Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")

    _, val_loader = get_dataloaders(config, world_size=1, rank=0)

    results = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Sampling")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            if args.num_samples and len(results) >= args.num_samples:
                break

            cond = batch['cond'].to(device)
            x_clean = batch['aps'].to(device)
            aps_pred = model(cond)

            for i in range(x_clean.shape[0]):
                results.append({'gt': x_clean[i].cpu().numpy(), 'pred': aps_pred[i].cpu().numpy()})
                if args.num_samples and len(results) >= args.num_samples:
                    break

    if args.num_samples and len(results) > args.num_samples:
        results = results[:args.num_samples]

    total_time = time.time() - start_time
    metrics = compute_metrics(results)

    print(f"\n=== RadioUNet 评估结果 ===")
    print(f"样本数: {len(results)}, 耗时: {total_time:.2f}s")
    print(f"MAE: {metrics['MAE']:.6f}, RMSE: {metrics['RMSE']:.6f}, PSNR: {metrics['PSNR']:.2f} dB, CosSim: {metrics['Cosine_Similarity']:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_json:
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()} | {'num_samples': len(results), 'time': total_time}, f, indent=2)

    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        for i in range(min(args.num_vis, len(results))):
            visualize_aps(results[i]['gt'], results[i]['pred'], vis_dir / f'sample_{i:03d}.png')
        print(f"可视化已保存到 {vis_dir}")

    np.savez(output_dir / 'results.npz', gt=[r['gt'] for r in results], pred=[r['pred'] for r in results])


if __name__ == '__main__':
    main()
