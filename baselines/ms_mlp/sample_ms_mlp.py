"""Sampling / evaluation script for MS-MLP (Ablation Baseline)"""
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import load_config, set_seed
from data.dataset import get_dataloaders
from baselines.ms_mlp.models import MSMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='baselines/ms_mlp/config_ms_mlp.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--gpu',        type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)

    config['model']['use_pretrained'] = False
    model = MSMLP(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, val_loader = get_dataloaders(config, world_size=1, rank=0)

    maes, cosines = [], []
    n = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            cond = batch['cond'].to(device)
            gt   = batch['aps'].to(device)
            pred = model(cond)
            for i in range(gt.shape[0]):
                g = gt[i].cpu().numpy()
                p = pred[i].cpu().numpy()
                maes.append(float(np.abs(g - p).mean()))
                cosines.append(float(np.dot(g, p) / (np.linalg.norm(g) * np.linalg.norm(p) + 1e-8)))
                n += 1
                if args.num_samples and n >= args.num_samples:
                    break
            if args.num_samples and n >= args.num_samples:
                break

    print(f"MS-MLP | n={n}, MAE={np.mean(maes):.6f}, CosSim={np.mean(cosines):.4f}")


if __name__ == '__main__':
    main()
