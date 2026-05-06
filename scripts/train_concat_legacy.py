#!/usr/bin/env python
"""Training entry point for the legacy concat UNet (L0).

Reuses the full Phase-1 training infrastructure (SliceDataset, frozen VAE,
fixed-val visualization, decoded MAE/PSNR/SSIM metrics, CBCT-init DDIM sampler,
EMA, SD/LDM warmup-constant scheduler) but swaps in ``UNetConcatLegacy``:
no ControlNet, no DR, no PACA, no region embedding.

The trainer (``train_unet_concat_control_paca``) is called with controlnet/DR
disabled — the legacy model's forward signature ignores the control kwargs.

Usage:
    python scripts/train_concat_legacy.py \
        --manifest data/manifest.csv \
        --vae-path checkpoints/vae/vae_best.pth \
        --save-dir checkpoints/phase1_matrix/L0-strong-bc256 \
        --base-channels 256 \
        --epochs 400 --early-stopping 40 \
        --lr 1e-4 --lr-schedule sd-warmup-constant --warmup-steps 10000 \
        --use-ema --ema-decay 0.9995 \
        --sampler-init cbct --sampler-t-start 300 --sampler-alpha 1.0 \
        --exp-id L0-strong --stage strong
"""
import os
import sys
import argparse
import random
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from models.vae import load_vae
from models.unetConcatLegacy import UNetConcatLegacy
from models.unetConcatControlPACA import train_unet_concat_control_paca
from utils.slice_dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train UNetConcatLegacy (L0)")

    # Required
    p.add_argument("--manifest", required=True, help="manifest CSV produced by preprocess_synthrad_dataset.py")
    p.add_argument("--vae-path", required=True, help="path to frozen VAE checkpoint (.pth)")

    # Output
    p.add_argument("--save-dir", default="checkpoints/concat_legacy",
                   help="directory for checkpoints and prediction samples")

    # Data
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment",  action="store_true",
                   help="disable training augmentation (flip + rotation)")

    # Model
    p.add_argument("--base-channels", type=int, default=256,
                   help="base channel width for legacy concat UNet")
    p.add_argument("--dropout-rate", type=float, default=0.1,
                   help="dropout rate for the UNet backbone (legacy default 0.1)")
    p.add_argument("--latent-mode", choices=["mu", "sample"], default="mu",
                   help="VAE latent used by diffusion training")

    # Training
    p.add_argument("--epochs",       type=int,   default=400)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int,   default=10000)
    p.add_argument("--lr-schedule", choices=["sd-warmup-constant", "constant"], default="sd-warmup-constant",
                   help="learning-rate schedule. sd-warmup-constant: linear warmup → constant")
    p.add_argument("--use-ema", action="store_true",
                   help="enable UNet EMA for validation, decoded metrics, and saved unet_ema.pth")
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--latent-stats-batches", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--no-amp", action="store_true",
                   help="disable CUDA AMP autocast and GradScaler")
    p.add_argument("--early-stopping", type=int, default=40,
                   help="stop if val loss doesn't improve for this many epochs")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="optional smoke-test limit for train batches per epoch")
    p.add_argument("--max-val-batches", type=int, default=None,
                   help="optional smoke-test limit for val batches per epoch")
    p.add_argument("--ddim-steps", type=int, default=50,
                   help="DDIM steps for decoded validation metrics and fixed-val images")
    p.add_argument("--sampler-init", choices=["noise", "cbct"], default="cbct",
                   help="initial latent for decoded validation sampling")
    p.add_argument("--sampler-t-start", type=int, default=300,
                   help="starting diffusion timestep for decoded validation sampling")
    p.add_argument("--sampler-alpha", type=float, default=1.0,
                   help="CBCT latent strength when --sampler-init=cbct")
    p.add_argument("--eval-every", type=int, default=10,
                   help="run decoded metrics and fixed-val image upload every N epochs")
    p.add_argument("--seed", type=int, default=42)

    # WandB
    p.add_argument("--no-wandb", action="store_true", help="disable WandB logging")
    p.add_argument("--wandb-project", type=str, default="cbct2sct_IBA",
                   help="WandB project name")
    p.add_argument("--wandb-name", type=str, default=None,
                   help="WandB run name")
    p.add_argument("--wandb-group", type=str, default="phase1-matrix-2026-05")
    p.add_argument("--exp-id", type=str, default="L0",
                   help="experiment id used in WandB name/tags, e.g. L0 / L0-strong")
    p.add_argument("--stage", type=str, default="strong",
                   choices=["smoke", "screen", "strong", "long", "manual"])
    p.add_argument("--fixed-val-config", type=str, default="configs/fixed_val_cases.yaml")
    p.add_argument("--fixed-val-cases-per-region", type=int, default=4)
    p.add_argument("--fixed-val-slices-per-case", type=int, default=3)
    p.add_argument("--fixed-val-max-images", type=int, default=48,
                   help="number of fixed validation slices kept for WandB visualization")

    # Resume
    p.add_argument("--unet-path", default=None,
                   help="resume: path to saved UNet state dict (unet_full.pth)")
    p.add_argument("--ema-path",  default=None,
                   help="resume: path to saved UNet EMA state dict (unet_ema_state.pth)")

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_short_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_fixed_val_batch(val_dataset, config_path, cases_per_region, slices_per_case, max_images):
    items = val_dataset.fixed_val_items(cases_per_region=cases_per_region, slices_per_case=slices_per_case)
    if not items:
        return None

    full_path = config_path if os.path.isabs(config_path) else os.path.join(PROJECT_ROOT, config_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if not os.path.exists(full_path):
        with open(full_path, "w") as f:
            f.write("fixed_val_cases:\n")
            current_key = None
            for item in items:
                key = (item["patient_id"], item["region"])
                if key != current_key:
                    f.write(f"  - patient_id: {item['patient_id']}\n")
                    f.write(f"    region: {item['region']}\n")
                    f.write("    slices:\n")
                    current_key = key
                f.write(f"      - {item['z']}\n")
        print(f"Fixed val config written: {full_path}")
    else:
        print(f"Fixed val config exists: {full_path}")

    items = items[:max_images]
    ct = torch.stack([x["ct"] for x in items])
    cbct = torch.stack([x["cbct"] for x in items])
    mask = torch.stack([x["mask"] for x in items])
    region_id = torch.stack([x["region_id"] for x in items])
    captions = [f"{x['patient_id']} {x['region']} z{x['z']}" for x in items]
    print(f"Fixed val visualized slices: {len(items)}")
    return ct, cbct, mask, region_id, captions


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = git_short_sha()
    run_name = args.wandb_name or f"{args.exp_id}-bc{args.base_channels}-ep{args.epochs}-s{args.seed}"

    print("=" * 60)
    print("UNetConcatLegacy (L0) Training")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device  : {device}")
    print("=" * 60)

    # ---- Data ---------------------------------------------------------------
    augmentation = not args.no_augment
    train_loader, val_loader = get_dataloaders(
        manifest_csv=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=augmentation,
        seed=args.seed,
    )
    print(f"Train slices : {len(train_loader.dataset)}")
    print(f"Val   slices : {len(val_loader.dataset)}")
    print(f"Augmentation : {'on' if augmentation else 'off'}")

    # ---- Models -------------------------------------------------------------
    vae = load_vae(save_path=args.vae_path, trainable=False)
    vae.eval()
    print(f"VAE loaded (frozen): {args.vae_path}")

    unet = UNetConcatLegacy(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        dropout_rate=args.dropout_rate,
    )
    if args.unet_path:
        state = torch.load(args.unet_path, map_location="cpu")
        missing, unexpected = unet.load_state_dict(state, strict=False)
        if missing:
            print(f"  UNet missing keys  : {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  UNet unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print(f"UNet resumed from: {args.unet_path}")

    n_unet = sum(p.numel() for p in unet.parameters())
    print(f"\nTrainable parameters:")
    print(f"  UNet (legacy concat): {n_unet/1e6:.1f}M")

    fixed_val_batch = build_fixed_val_batch(
        val_loader.dataset,
        args.fixed_val_config,
        args.fixed_val_cases_per_region,
        args.fixed_val_slices_per_case,
        args.fixed_val_max_images,
    )

    # ---- Train --------------------------------------------------------------
    save_dir    = os.path.join(PROJECT_ROOT, args.save_dir)
    predict_dir = os.path.join(save_dir, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)

    print(f"\nCheckpoints → {save_dir}")
    print(
        f"Run: {run_name}  Commit: {commit}\n"
        f"Epochs: {args.epochs}  LR: {args.lr}  wd: {args.weight_decay}  "
        f"lr-schedule: {args.lr_schedule} warmup: {args.warmup_steps}  "
        f"dropout: {args.dropout_rate}  "
        f"EMA: {'on' if args.use_ema else 'off'}  "
        f"grad-accum: {args.grad_accum_steps}  AMP: {'off' if args.no_amp else 'on'}  "
        f"early-stop: {args.early_stopping}\n"
        f"Model: legacy concat (no ControlNet, no DR, no PACA, no region embedding)"
    )
    print("=" * 60 + "\n")

    wandb_logger = None
    if not args.no_wandb:
        from utils.wandb_logger import WandbLogger
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config={
                "exp_id": args.exp_id,
                "stage": args.stage,
                "commit": commit,
                "manifest": args.manifest,
                "vae_path": args.vae_path,
                "save_dir": args.save_dir,
                "model": "UNetConcatLegacy",
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "augmentation": augmentation,
                "base_channels": args.base_channels,
                "dropout_rate": args.dropout_rate,
                "epochs": args.epochs,
                "lr": args.lr,
                "lr_schedule": args.lr_schedule,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
                "latent_stats_batches": args.latent_stats_batches,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch": args.batch_size * args.grad_accum_steps,
                "amp": not args.no_amp,
                "seed": args.seed,
                "use_dr": False,
                "use_controlnet": False,
                "controlnet_fusion": "none",
                "control_source": "none",
                "region_embedding": False,
                "latent_mode": args.latent_mode,
                "ddim_steps": args.ddim_steps,
                "sampler_init": args.sampler_init,
                "sampler_t_start": args.sampler_t_start,
                "sampler_alpha": args.sampler_alpha,
                "eval_every": args.eval_every,
                "early_stopping": args.early_stopping,
                "max_train_batches": args.max_train_batches,
                "max_val_batches": args.max_val_batches,
                "train_slices": len(train_loader.dataset),
                "val_slices": len(val_loader.dataset),
            },
            tags=["legacy-concat", "phase1", args.exp_id, args.stage, commit],
            group=args.wandb_group,
            notes="UNetConcatLegacy (L0) training — legacy concat baseline retrain",
        )

    try:
        train_unet_concat_control_paca(
            vae=vae,
            unet=unet,
            controlnet=None,
            dr_module=None,
            control_adapter=None,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_dir=save_dir,
            predict_dir=predict_dir,
            early_stopping=args.early_stopping,
            gamma=0.0,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            lr_schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            ema_path=args.ema_path,
            latent_stats_batches=args.latent_stats_batches,
            wandb_logger=wandb_logger,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            grad_accum_steps=args.grad_accum_steps,
            amp_enabled=torch.cuda.is_available() and not args.no_amp,
            use_dr=False,
            use_controlnet=False,
            control_source="cbct_latent",
            controlnet_fusion="add",
            latent_mode=args.latent_mode,
            ddim_steps=args.ddim_steps,
            sampler_init=args.sampler_init,
            sampler_t_start=args.sampler_t_start,
            sampler_alpha=args.sampler_alpha,
            eval_every=args.eval_every,
            fixed_val_batch=fixed_val_batch,
            fixed_val_max_images=args.fixed_val_max_images,
        )
    finally:
        if wandb_logger:
            wandb_logger.finish()

    print("\n" + "=" * 60)
    print(f"Training finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoints saved in: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
