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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from models.vae import VAE
from utils.losses import PerceptualLoss, SsimLoss
from utils.slice_dataset import SliceDataset, REGION_TO_ID
from utils.wandb_logger import WandbLogger
from utils.manifest_checks import validate_manifest_clip
from utils.hu import CLIP_MIN, CLIP_MAX, to_hu
from utils.image_metrics import ImageMetrics

REGION_NAMES = {v: k for k, v in REGION_TO_ID.items()}


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


def _image01(x: torch.Tensor):
    return ((x.detach().float().cpu().squeeze().clamp(-1, 1) + 1.0) / 2.0).numpy()


def log_vae_images(vae, fixed_batch, device, epoch, wandb_logger, amp_enabled, max_samples, prefix):
    """Log fixed reconstructions for stable visual monitoring."""
    if not wandb_logger or fixed_batch is None:
        return

    vae.eval()
    x = fixed_batch[:max_samples].to(device, non_blocking=True)
    with torch.no_grad():
        with autocast(enabled=amp_enabled):
            _, _, _, recon = vae(x)

    n = min(max_samples, x.shape[0])
    for i in range(n):
        original = _image01(x[i])
        reconstructed = _image01(recon[i])
        error = torch.abs(recon[i].detach().float() - x[i].detach().float()).cpu().squeeze().clamp(0, 2).numpy() / 2.0
        wandb_logger.log_image(f"{prefix}_visual/original_{i}", original, caption=f"epoch {epoch} {prefix} original", step=epoch)
        wandb_logger.log_image(f"{prefix}_visual/reconstructed_{i}", reconstructed, caption=f"epoch {epoch} {prefix} reconstructed", step=epoch)
        wandb_logger.log_image(f"{prefix}_visual/error_{i}", error, caption=f"epoch {epoch} {prefix} absolute error", step=epoch)


def validate_vae(vae, val_loader, device, perceptual_loss, ssim_loss, args, amp_enabled):
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

    avg_val = val_total / max(val_batches, 1)
    val_metrics = {f"val/{k.split('/')[-1]}": v / max(val_batches, 1) for k, v in val_parts.items()}
    return avg_val, val_metrics, val_batches


@torch.no_grad()
def patient_level_vae_metrics(vae, val_ds, device, batch_size: int, max_patients=None) -> dict:
    """Patient-level mae_hu / psnr / ms_ssim using SynthRAD ImageMetrics.

    Mirrors `_eval_decoded_metrics` in unetConcatControlPACA.py so VAE training
    curves are on the same scale as D1 training curves and the leaderboard.
    """
    metrics_obj = ImageMetrics()
    slice_ds = val_ds.slice_ds  # underlying SliceDataset

    n_vols = len(slice_ds._vols)
    if max_patients is not None:
        n_vols = min(n_vols, int(max_patients))

    totals = {"mae": 0.0, "psnr": 0.0, "ms_ssim": 0.0}
    region_totals = {name: {"mae": 0.0, "psnr": 0.0, "ms_ssim": 0.0, "n": 0}
                     for name in REGION_NAMES.values()}
    n_patients = 0

    vae.eval()
    for vi in range(n_vols):
        ct_full, _cbct, mask_full, rid = slice_ds._vols[vi]
        Z = ct_full.shape[0]
        recon_full = np.empty_like(ct_full)
        for z0 in range(0, Z, batch_size):
            z1 = min(z0 + batch_size, Z)
            ct_b = torch.from_numpy(ct_full[z0:z1]).unsqueeze(1).to(device)
            mu, _ = vae.encode(ct_b)
            recon = vae.decode(mu).clamp(-1.0, 1.0)
            recon_full[z0:z1] = recon.squeeze(1).detach().cpu().numpy()

        body = (mask_full > 0.5)
        recon_full = recon_full * body + (-1.0) * (1.0 - body)

        ct_hu  = to_hu(ct_full)
        rec_hu = to_hu(recon_full)
        scored = metrics_obj.score_patient(ct_hu, rec_hu, mask_full)

        totals["mae"]     += scored["mae"]
        totals["psnr"]    += scored["psnr"]
        totals["ms_ssim"] += scored["ms_ssim"]
        n_patients += 1

        name = REGION_NAMES[int(rid)]
        region_totals[name]["mae"]     += scored["mae"]
        region_totals[name]["psnr"]    += scored["psnr"]
        region_totals[name]["ms_ssim"] += scored["ms_ssim"]
        region_totals[name]["n"]       += 1

    n_safe = max(n_patients, 1)
    out = {
        "val/mae_hu_vae":  totals["mae"]     / n_safe,
        "val/psnr_vae":    totals["psnr"]    / n_safe,
        "val/ms_ssim_vae": totals["ms_ssim"] / n_safe,
        "val/n_patients":  float(n_patients),
    }
    for name, t in region_totals.items():
        if t["n"]:
            out[f"val/mae_hu_vae_{name}"]  = t["mae"]     / t["n"]
            out[f"val/psnr_vae_{name}"]    = t["psnr"]    / t["n"]
            out[f"val/ms_ssim_vae_{name}"] = t["ms_ssim"] / t["n"]
    return out


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
    p.add_argument("--eval-only", action="store_true",
                   help="load --resume and run validation/visualization without training")
    p.add_argument("--start-epoch", type=int, default=0,
                   help="number of completed epochs when resuming; logging continues from start_epoch + 1")
    p.add_argument("--initial-best-val", type=float, default=None,
                   help="previous best validation loss when resuming")
    p.add_argument("--perceptual-weight", type=float, default=0.1)
    p.add_argument("--ssim-weight", type=float, default=0.8)
    p.add_argument("--mse-weight", type=float, default=0.0)
    p.add_argument("--kl-weight", type=float, default=1e-5)
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None,
                   help="cap val batches for the loss-only val pass (smoke test)")
    p.add_argument("--max-val-patients", type=int, default=None,
                   help="cap patients for the patient-level HU eval; None = full val (~23)")
    p.add_argument("--vis-every", type=int, default=10,
                   help="upload fixed train/validation reconstructions to WandB every N epochs; 0 disables")
    p.add_argument("--vis-num-samples", type=int, default=4,
                   help="number of fixed train and validation samples to visualize")
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

    # Fail-fast if data was preprocessed with a different HU clip range.
    validate_manifest_clip(args.manifest, "train", CLIP_MIN, CLIP_MAX)
    validate_manifest_clip(args.manifest, "val",   CLIP_MIN, CLIP_MAX)

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

    fixed_train_batch = None
    fixed_val_batch = None
    if wandb_logger and args.vis_every and args.vis_every > 0:
        fixed_train_batch = next(iter(train_loader))[:args.vis_num_samples].contiguous()
        fixed_val_batch = next(iter(val_loader))[:args.vis_num_samples].contiguous()
        print(
            f"Fixed visualization batches every {args.vis_every} epochs | "
            f"train {tuple(fixed_train_batch.shape)} | val {tuple(fixed_val_batch.shape)}"
        )

    print(f"Train slices : {len(train_ds)}")
    print(f"Val slices   : {len(val_ds)}")
    print(f"Batch size   : {args.batch_size}")
    print(f"AMP enabled  : {amp_enabled}")
    print(f"Save dir     : {args.save_dir}")

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume")
        avg_val, val_metrics, val_batches = validate_vae(
            vae=vae,
            val_loader=val_loader,
            device=device,
            perceptual_loss=perceptual_loss,
            ssim_loss=ssim_loss,
            args=args,
            amp_enabled=amp_enabled,
        )
        print(f"Eval only | Val {avg_val:.6f} | batches {val_batches}")
        if wandb_logger:
            wandb_logger.log_metrics(
                {
                    "epoch": args.start_epoch,
                    "val_loss": avg_val,
                    "val/batches": val_batches,
                    **val_metrics,
                },
                step=args.start_epoch,
            )
            log_vae_images(
                vae=vae,
                fixed_batch=fixed_val_batch,
                device=device,
                epoch=args.start_epoch,
                wandb_logger=wandb_logger,
                amp_enabled=amp_enabled,
                max_samples=args.vis_num_samples,
                prefix="val",
            )
            wandb_logger.finish()
        return

    best_val = args.initial_best_val if args.initial_best_val is not None else float("inf")
    bad_epochs = 0
    best_path = os.path.join(args.save_dir, "vae_best.pth")
    last_path = os.path.join(args.save_dir, "vae_last.pth")

    try:
        for epoch in range(args.start_epoch, args.epochs):
            epoch_num = epoch + 1
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
                f"Epoch {epoch_num}/{args.epochs} | "
                f"Train {avg_train:.6f} | Val {avg_val:.6f} | LR {lr:.2e}"
            )

            if wandb_logger:
                wandb_logger.log_metrics(
                    {
                        "epoch": epoch_num,
                        "train_loss": avg_train,
                        "val_loss": avg_val,
                        "learning_rate": lr,
                        "train/batches": train_batches,
                        "val/batches": val_batches,
                        **train_metrics,
                        **val_metrics,
                    },
                    step=epoch_num,
                )

            torch.save(vae.state_dict(), last_path)
            if avg_val < best_val:
                best_val = avg_val
                bad_epochs = 0
                torch.save(vae.state_dict(), best_path)
                print(f"Saved best VAE: {best_path} (val {best_val:.6f})")
            else:
                bad_epochs += 1

            if wandb_logger and args.vis_every and args.vis_every > 0 and epoch_num % args.vis_every == 0:
                log_vae_images(
                    vae=vae,
                    fixed_batch=fixed_train_batch,
                    device=device,
                    epoch=epoch_num,
                    wandb_logger=wandb_logger,
                    amp_enabled=amp_enabled,
                    max_samples=args.vis_num_samples,
                    prefix="train",
                )
                log_vae_images(
                    vae=vae,
                    fixed_batch=fixed_val_batch,
                    device=device,
                    epoch=epoch_num,
                    wandb_logger=wandb_logger,
                    amp_enabled=amp_enabled,
                    max_samples=args.vis_num_samples,
                    prefix="val",
                )
                # Patient-level SynthRAD-aligned metrics for tracking the HU-space floor.
                hu_metrics = patient_level_vae_metrics(
                    vae=vae,
                    val_ds=val_ds,
                    device=device,
                    batch_size=args.batch_size,
                    max_patients=args.max_val_patients,
                )
                wandb_logger.log_metrics({"epoch": epoch_num, **hu_metrics}, step=epoch_num)
                print(
                    f"  [hu] mae_hu_vae={hu_metrics['val/mae_hu_vae']:.2f}  "
                    f"psnr_vae={hu_metrics['val/psnr_vae']:.2f}  "
                    f"ms_ssim_vae={hu_metrics['val/ms_ssim_vae']:.4f}  "
                    f"(n={int(hu_metrics['val/n_patients'])})"
                )

            if args.early_stopping and bad_epochs >= args.early_stopping:
                print(f"Early stopped after {args.early_stopping} epochs without improvement.")
                break
    finally:
        if wandb_logger:
            wandb_logger.finish()

    print(f"Training finished. Best val: {best_val:.6f}")


if __name__ == "__main__":
    main()
