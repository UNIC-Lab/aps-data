"""Training script for Adv-MLP Baseline"""
import os
import sys
import torch
import torch.nn as nn
from torch.amp import autocast
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import load_config, set_seed
from data.dataset import get_dataloaders
from baselines.adv_mlp.models import AdvMLP


def setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_adv_mlp_{timestamp}.log"
    logger = logging.getLogger('train_adv_mlp')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_lr_scheduler(optimizer, config, steps_per_epoch):
    warmup_steps = config['training'].get('warmup_steps', 500)
    total_steps = config['training']['epochs'] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baselines/adv_mlp/config_adv_mlp.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    set_seed(config['seed'])
    logger = setup_logger(config['logging'].get('log_dir', 'baselines/adv_mlp/logs'))

    model = AdvMLP(config).cuda()
    G = model.generator
    D = model.discriminator

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    logger.info(f"Generator: {g_params:,}, Discriminator: {d_params:,}, Total: {g_params + d_params:,}")

    # Optimizers for adversarial training.
    opt_g = torch.optim.Adam(G.parameters(), lr=config['training']['lr_g'],
                             betas=(config['training']['beta1'], config['training']['beta2']))
    opt_d = torch.optim.Adam(D.parameters(), lr=config['training']['lr_d'],
                             betas=(config['training']['beta1'], config['training']['beta2']))

    train_loader, val_loader = get_dataloaders(config, world_size=1, rank=0)

    sched_g = get_lr_scheduler(opt_g, config, len(train_loader))
    sched_d = get_lr_scheduler(opt_d, config, len(train_loader))

    use_amp = config['training']['mixed_precision'] in ['bf16', 'fp16']
    dtype = torch.bfloat16 if config['training']['mixed_precision'] == 'bf16' else torch.float16

    lambda_l1 = config['training']['lambda_l1']
    n_critic = config['training']['n_critic']
    criterion = nn.BCEWithLogitsLoss()

    if config['logging']['use_wandb']:
        wandb.init(project=config['logging']['wandb_project'], name=config['logging']['exp_name'], config=config)

    best_val_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        G.train()
        D.train()
        total_g_loss = 0
        total_d_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            x_real = batch['aps'].cuda()        # (B, 180)
            cond = batch['cond'].cuda()          # (B, 3, H, W)
            B = x_real.shape[0]

            real_label = torch.ones(B, 1, device=x_real.device)
            fake_label = torch.zeros(B, 1, device=x_real.device)

            # ===== Train Discriminator =====
            with autocast('cuda', dtype=dtype, enabled=use_amp):
                x_fake = G(cond).detach()
                d_real = D(cond, x_real)
                d_fake = D(cond, x_fake)
                loss_d = criterion(d_real, real_label) + criterion(d_fake, fake_label)

            opt_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), config['training']['grad_clip'])
            opt_d.step()
            sched_d.step()

            # ===== Train Generator =====
            if (batch_idx + 1) % n_critic == 0:
                with autocast('cuda', dtype=dtype, enabled=use_amp):
                    x_fake = G(cond)
                    d_fake = D(cond, x_fake)
                    loss_g_adv = criterion(d_fake, real_label)
                    loss_g_l1 = nn.functional.l1_loss(x_fake, x_real)
                    loss_g = loss_g_adv + lambda_l1 * loss_g_l1

                opt_g.zero_grad()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), config['training']['grad_clip'])
                opt_g.step()
                sched_g.step()

                total_g_loss += loss_g.item()

            total_d_loss += loss_d.item()

            pbar.set_postfix({
                'D': f"{loss_d.item():.4f}",
                'G': f"{loss_g.item():.4f}" if (batch_idx + 1) % n_critic == 0 else '-'
            })

        avg_d = total_d_loss / len(train_loader)
        avg_g = total_g_loss / (len(train_loader) // n_critic)

        # Validate (G only, MSE)
        G.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_clean = batch['aps'].cuda()
                cond = batch['cond'].cuda()
                # 推理时不加噪声
                aps_pred = G(cond, z=torch.zeros(x_clean.shape[0], model.z_dim, device=x_clean.device))
                val_loss += nn.functional.mse_loss(aps_pred, x_clean).item()
        val_loss /= len(val_loader)

        log_msg = f"Epoch {epoch}: D={avg_d:.4f}, G={avg_g:.4f}, val_mse={val_loss:.4f}"
        print(log_msg)
        logger.info(log_msg)

        if config['logging']['use_wandb']:
            wandb.log({'epoch': epoch, 'train/d_loss': avg_d, 'train/g_loss': avg_g, 'val/mse': val_loss})

        ckpt_dir = Path(config['logging']['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if epoch % 2 == 0 or epoch == config['training']['epochs']:
            torch.save({'epoch': epoch, 'g_state_dict': G.state_dict(), 'd_state_dict': D.state_dict(),
                        'val_loss': val_loss, 'config': config}, ckpt_dir / f'epoch_{epoch}.pt')

        total_epochs = config['training']['epochs']
        if epoch >= total_epochs // 2 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'g_state_dict': G.state_dict(), 'd_state_dict': D.state_dict(),
                        'val_loss': val_loss, 'config': config}, ckpt_dir / 'best_model.pt')
            print(f"保存最佳模型 (val={val_loss:.4f})")

    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == '__main__':
    main()
