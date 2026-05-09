"""Sampling script for ResNet-Conv1D Baseline"""
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
from baselines.resnet_conv1d.models import ResNetConv1D


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
        'MAE': np.mean(maes), 'MSE': np.mean(mses), 'RMSE': np.sqrt(np.mean(mses)),
        'PSNR': np.mean(psnrs), 'Cosine_Similarity': np.mean(cosine_sims),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baselines/resnet_conv1d/config_resnet_conv1d.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='baselines/resnet_conv1d/samples')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_vis', type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)

    config['model']['use_pretrained'] = False
    model = ResNetConv1D(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, val_loader = get_dataloaders(config, world_size=1, rank=0)
    results = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Sampling"):
            if args.num_samples and len(results) >= args.num_samples:
                break
            cond = batch['cond'].to(device)
            gt = batch['aps'].to(device)
            pred = model(cond)
            for i in range(gt.shape[0]):
                results.append({'gt': gt[i].cpu().numpy(), 'pred': pred[i].cpu().numpy()})
                if args.num_samples and len(results) >= args.num_samples:
                    break
    if args.num_samples:
        results = results[:args.num_samples]

    metrics = compute_metrics(results)
    print(f"ResNet-Conv1D | MAE={metrics['MAE']:.6f}, PSNR={metrics['PSNR']:.2f}, CosSim={metrics['Cosine_Similarity']:.4f}")

    if args.visualize:
        output_dir = Path(args.output_dir)
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(args.num_vis, len(results))):
            visualize_aps(results[i]['gt'], results[i]['pred'], vis_dir / f'sample_{i:03d}.png')


if __name__ == '__main__':
    main()
