"""Training script for MS-MLP (Ablation Baseline)

Multi-Scale Encoder + MLP Decoder, pure MSE regression.
No adversarial regularization. Isolates multi-scale encoding contribution.
"""
import sys
import torch
import torch.nn as nn
from torch.amp import autocast
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import load_config, set_seed
from data.dataset import get_dataloaders
from baselines.ms_mlp.models import MSMLP


def setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_ms_mlp_{timestamp}.log"
    logger = logging.getLogger('train_ms_mlp')
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
    warmup_steps = config['training']['warmup_steps']
    total_steps = config['training']['epochs'] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baselines/ms_mlp/config_ms_mlp.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    gpu_id = args.gpu if args.gpu is not None else 0
    torch.cuda.set_device(gpu_id)

    set_seed(config['seed'])
    logger = setup_logger(config['logging'].get('log_dir', 'baselines/ms_mlp/logs'))

    torch.backends.cudnn.benchmark = True

    model_raw = MSMLP(config).cuda()
    model = torch.compile(model_raw)

    total = sum(p.numel() for p in model_raw.parameters())
    logger.info(f"MS-MLP 参数: {total:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    train_loader, val_loader = get_dataloaders(config, world_size=1, rank=0)
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))

    use_amp = config['training']['mixed_precision'] in ['bf16', 'fp16']
    dtype = torch.bfloat16 if config['training']['mixed_precision'] == 'bf16' else torch.float16

    best_val_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            x_clean = batch['aps'].cuda()
            cond = batch['cond'].cuda()
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=dtype, enabled=use_amp):
                loss = nn.functional.mse_loss(model(cond), x_clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                x_clean = batch['aps'].cuda()
                cond = batch['cond'].cuda()
                with autocast('cuda', dtype=dtype, enabled=use_amp):
                    val_loss += nn.functional.mse_loss(model(cond), x_clean).item()
        val_loss /= len(val_loader)

        msg = f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}"
        print(msg)
        logger.info(msg)

        ckpt_dir = Path(config['logging']['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if epoch % 2 == 0 or epoch == config['training']['epochs']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_raw.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, ckpt_dir / f'epoch_{epoch}.pt')

        total_epochs = config['training']['epochs']
        if epoch >= total_epochs // 2 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_raw.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, ckpt_dir / 'best_model.pt')
            logger.info(f"  best model saved (val={val_loss:.4f})")


if __name__ == '__main__':
    main()
