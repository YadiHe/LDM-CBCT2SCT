#!/usr/bin/env python
"""
Train the CT VAE on preprocessed MHA volumes from data/manifest.csv.

This is the Phase 0 prerequisite for ConcatPACA Phase 1 training. The dataset
uses all CT slices from synthRAD2023/2025 preprocessed outputs.
"""
import os
import sys
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from models.vae import VAE
from utils.losses import PerceptualLoss, SsimLoss
from utils.slice_dataset import SliceDataset
from utils.wandb_logger import WandbLogger


class CTOnlyDataset(Dataset):
    """Expose CT images only from SliceDataset."""

    def __init__(self, manifest_csv: str, split: str, augmentation: bool = False):
        self.slice_ds = SliceDataset(manifest_csv, split, augmentation=augmentation)

    def __len__(self):
        return len(self.slice_ds)

    def __getitem__(self, idx):
        ct, _, _, _ = self.slice_ds[idx]
        return ct


def vae_loss_components(
    recon,
    x,
    mu,
    logvar,
    perceptual_loss,
    ssim_loss,
    perceptual_weight=0.1,
    ssim_weight=0.8,
    mse_weight=0.0,
    kl_weight=1e-5,
    l1_weight=1.0,
):
    mse = F.mse_loss(recon, x)
    l1 = F.l1_loss(recon, x)
    kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]))
    ssim_val = ssim_loss(recon, x)
    perceptual = perceptual_loss(recon, x) if perceptual_weight > 0 else recon.new_tensor(0.0)
    total = (
        mse_weight * mse
        + l1_weight * l1
        + kl_weight * kl
        + ssim_weight * ssim_val
        + perceptual_weight * perceptual
    )
    return total, {
        "loss/mse": mse.item(),
        "loss/l1": l1.item(),
        "loss/kl": kl.item(),
        "loss/ssim": ssim_val.item(),
        "loss/perceptual": perceptual.item(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE on preprocessed CT slices")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--save-dir", default="checkpoints/vae")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=6.25e-6)
    p.add_argument("--early-stopping", type=int, default=30)
    p.add_argument("--patience", type=int, default=10, help="LR scheduler patience")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--latent-channels", type=int, default=3)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--perceptual-weight", type=float, default=0.1)
    p.add_argument("--ssim-weight", type=float, default=0.8)
    p.add_argument("--mse-weight", type=float, default=0.0)
    p.add_argument("--kl-weight", type=float, default=1e-5)
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="cbct2sct_IBA")
    p.add_argument("--wandb-name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Preprocessed CT VAE Training")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device  : {device}")
    print("=" * 60)

    augmentation = not args.no_augment
    train_ds = CTOnlyDataset(args.manifest, "train", augmentation=augmentation)
    val_ds = CTOnlyDataset(args.manifest, "val", augmentation=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    vae = VAE(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
    ).to(device)
    if args.resume:
        vae.load_state_dict(torch.load(args.resume, map_location=device), strict=True)
        print(f"Resumed VAE from: {args.resume}")

    perceptual_loss = PerceptualLoss(device=device)
    ssim_loss = SsimLoss()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.patience,
        threshold=1e-4,
        min_lr=1e-6,
    )
    amp_enabled = torch.cuda.is_available() and not args.no_amp
    scaler = GradScaler(enabled=amp_enabled)

    wandb_logger = None
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=["vae", "preprocessed-mha", "phase0"],
            notes="Full CT VAE training on synthRAD2023/2025 preprocessed slices",
        )

    print(f"Train slices : {len(train_ds)}")
    print(f"Val slices   : {len(val_ds)}")
    print(f"Batch size   : {args.batch_size}")
    print(f"AMP enabled  : {amp_enabled}")
    print(f"Save dir     : {args.save_dir}")

    best_val = float("inf")
    bad_epochs = 0
    best_path = os.path.join(args.save_dir, "vae_best.pth")
    last_path = os.path.join(args.save_dir, "vae_last.pth")

    try:
        for epoch in range(args.epochs):
            vae.train()
            train_total = 0.0
            train_parts = {}
            train_batches = 0
            for i, x in enumerate(train_loader):
                if args.max_train_batches is not None and i >= args.max_train_batches:
                    break
                x = x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=amp_enabled):
                    _, mu, logvar, recon = vae(x)
                    loss, parts = vae_loss_components(
                        recon,
                        x,
                        mu,
                        logvar,
                        perceptual_loss,
                        ssim_loss,
                        args.perceptual_weight,
                        args.ssim_weight,
                        args.mse_weight,
                        args.kl_weight,
                        args.l1_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_total += loss.item()
                for k, v in parts.items():
                    train_parts[k] = train_parts.get(k, 0.0) + v
                train_batches += 1

            vae.eval()
            val_total = 0.0
            val_parts = {}
            val_batches = 0
            with torch.no_grad():
                for i, x in enumerate(val_loader):
                    if args.max_val_batches is not None and i >= args.max_val_batches:
                        break
                    x = x.to(device, non_blocking=True)
                    with autocast(enabled=amp_enabled):
                        _, mu, logvar, recon = vae(x)
                        loss, parts = vae_loss_components(
                            recon,
                            x,
                            mu,
                            logvar,
                            perceptual_loss,
                            ssim_loss,
                            args.perceptual_weight,
                            args.ssim_weight,
                            args.mse_weight,
                            args.kl_weight,
                            args.l1_weight,
                        )
                    val_total += loss.item()
                    for k, v in parts.items():
                        val_parts[k] = val_parts.get(k, 0.0) + v
                    val_batches += 1

            avg_train = train_total / max(train_batches, 1)
            avg_val = val_total / max(val_batches, 1)
            scheduler.step(avg_val)
            lr = optimizer.param_groups[0]["lr"]

            train_metrics = {f"train/{k.split('/')[-1]}": v / max(train_batches, 1) for k, v in train_parts.items()}
            val_metrics = {f"val/{k.split('/')[-1]}": v / max(val_batches, 1) for k, v in val_parts.items()}
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train {avg_train:.6f} | Val {avg_val:.6f} | LR {lr:.2e}"
            )

            if wandb_logger:
                wandb_logger.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train,
                        "val_loss": avg_val,
                        "learning_rate": lr,
                        "train/batches": train_batches,
                        "val/batches": val_batches,
                        **train_metrics,
                        **val_metrics,
                    },
                    step=epoch + 1,
                )

            torch.save(vae.state_dict(), last_path)
            if avg_val < best_val:
                best_val = avg_val
                bad_epochs = 0
                torch.save(vae.state_dict(), best_path)
                print(f"Saved best VAE: {best_path} (val {best_val:.6f})")
            else:
                bad_epochs += 1

            if args.early_stopping and bad_epochs >= args.early_stopping:
                print(f"Early stopped after {args.early_stopping} epochs without improvement.")
                break
    finally:
        if wandb_logger:
            wandb_logger.finish()

    print(f"Training finished. Best val: {best_val:.6f}")


if __name__ == "__main__":
    main()
