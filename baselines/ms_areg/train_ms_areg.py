"""Training script for MS-AReg (Proposed Method)

Regression with multi-scale encoder and adversarial regularization.
Training procedure identical to Adv-MLP; only the predictor encoder is upgraded.
"""
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
from baselines.ms_areg.models import MSAReg


def setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_ms_areg_{timestamp}.log"
    logger = logging.getLogger('train_ms_areg')
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
    parser.add_argument('--config', type=str, default='baselines/ms_areg/config_ms_areg.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    set_seed(config['seed'])
    logger = setup_logger(config['logging'].get('log_dir', 'baselines/ms_areg/logs'))

    torch.backends.cudnn.benchmark = True

    model = MSAReg(config).cuda()
    P_raw = model.predictor   # 保留原始引用，用于保存 state_dict
    R_raw = model.regularizer

    # H100 torch.compile
    P = torch.compile(P_raw)
    R = torch.compile(R_raw)

    p_params = sum(p.numel() for p in P_raw.parameters())
    r_params = sum(p.numel() for p in R_raw.parameters())
    logger.info(f"MS-AReg | Predictor: {p_params:,}, Regularizer: {r_params:,}, Total: {p_params + r_params:,}")

    opt_p = torch.optim.Adam(P.parameters(), lr=config['training']['lr_g'],
                             betas=(config['training']['beta1'], config['training']['beta2']))
    opt_r = torch.optim.Adam(R.parameters(), lr=config['training']['lr_d'],
                             betas=(config['training']['beta1'], config['training']['beta2']))

    train_loader, val_loader = get_dataloaders(config, world_size=1, rank=0)

    sched_p = get_lr_scheduler(opt_p, config, len(train_loader))
    sched_r = get_lr_scheduler(opt_r, config, len(train_loader))

    use_amp = config['training']['mixed_precision'] in ['bf16', 'fp16']
    dtype = torch.bfloat16 if config['training']['mixed_precision'] == 'bf16' else torch.float16

    lambda_l1 = config['training']['lambda_l1']
    n_critic = config['training']['n_critic']
    criterion = nn.BCEWithLogitsLoss()

    if config['logging']['use_wandb']:
        wandb.init(project=config['logging']['wandb_project'],
                   name=config['logging']['exp_name'], config=config)

    best_val_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        P.train()
        R.train()
        total_p_loss = 0
        total_r_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            x_real = batch['aps'].cuda()
            cond = batch['cond'].cuda()
            B = x_real.shape[0]

            real_label = torch.ones(B, 1, device=x_real.device)
            fake_label = torch.zeros(B, 1, device=x_real.device)

            # ===== Train Regularizer =====
            with autocast('cuda', dtype=dtype, enabled=use_amp):
                x_pred = P(cond).detach()
                r_real = R(cond, x_real)
                r_fake = R(cond, x_pred)
                loss_r = criterion(r_real, real_label) + criterion(r_fake, fake_label)

            opt_r.zero_grad(set_to_none=True)
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(R.parameters(), config['training']['grad_clip'])
            opt_r.step()
            sched_r.step()

            # ===== Train Predictor =====
            if (batch_idx + 1) % n_critic == 0:
                with autocast('cuda', dtype=dtype, enabled=use_amp):
                    x_pred = P(cond)
                    r_fake = R(cond, x_pred)
                    loss_adv = criterion(r_fake, real_label)
                    loss_l1 = nn.functional.l1_loss(x_pred, x_real)
                    loss_p = loss_adv + lambda_l1 * loss_l1

                opt_p.zero_grad(set_to_none=True)
                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(P.parameters(), config['training']['grad_clip'])
                opt_p.step()
                sched_p.step()

                total_p_loss += loss_p.item()

            total_r_loss += loss_r.item()

            pbar.set_postfix({
                'R': f"{loss_r.item():.4f}",
                'P': f"{loss_p.item():.4f}" if (batch_idx + 1) % n_critic == 0 else '-'
            })

        avg_r = total_r_loss / len(train_loader)
        avg_p = total_p_loss / (len(train_loader) // n_critic)

        # Validate (deterministic regression)
        P.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_clean = batch['aps'].cuda()
                cond = batch['cond'].cuda()
                aps_pred = P(cond)
                val_loss += nn.functional.mse_loss(aps_pred, x_clean).item()
        val_loss /= len(val_loader)

        log_msg = f"Epoch {epoch}: R={avg_r:.4f}, P={avg_p:.4f}, val_mse={val_loss:.4f}"
        print(log_msg)
        logger.info(log_msg)

        if config['logging']['use_wandb']:
            wandb.log({'epoch': epoch, 'train/r_loss': avg_r, 'train/p_loss': avg_p, 'val/mse': val_loss})

        # Save checkpoints
        ckpt_dir = Path(config['logging']['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if epoch % 2 == 0 or epoch == config['training']['epochs']:
            torch.save({
                'epoch': epoch, 'p_state_dict': P_raw.state_dict(), 'r_state_dict': R_raw.state_dict(),
                'val_loss': val_loss, 'config': config,
            }, ckpt_dir / f'epoch_{epoch}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'p_state_dict': P_raw.state_dict(), 'r_state_dict': R_raw.state_dict(),
                'val_loss': val_loss, 'config': config,
            }, ckpt_dir / 'best_model.pt')
            print(f"  best model saved (val_mse={val_loss:.4f})")

    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == '__main__':
    main()
