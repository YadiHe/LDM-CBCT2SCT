#!/usr/bin/env python
"""
训练入口：UNetConcatControlPACA + ControlNet + DegradationRemoval

两阶段流程中的第二阶段——冻结 VAE，联合训练 DR + ControlNet + UNet。
数据：SliceDataset（从 manifest CSV 加载预处理后的 MHA volume）

用法:
    python scripts/train_concat_paca.py \
        --manifest  data/manifest.csv \
        --vae-path  checkpoints/vae/vae_best.pth \
        --save-dir  checkpoints/concat_paca \
        --batch-size 16 \
        --grad-accum-steps 1 \
        --epochs 300 \
        --lr 5e-6

断点续训:
    python scripts/train_concat_paca.py \
        --manifest  data/manifest.csv \
        --vae-path  checkpoints/vae/vae_best.pth \
        --save-dir  checkpoints/concat_paca \
        --unet-path       checkpoints/concat_paca/unet_full.pth \
        --controlnet-path checkpoints/concat_paca/controlnet.pth \
        --dr-path         checkpoints/concat_paca/dr_module.pth
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
from models.controlnet import ControlNet
from models.degradationRemoval import DegradationRemoval
from models.unetConcatControlPACA import LatentControlAdapter, UNetConcatControlPACA, train_unet_concat_control_paca
from utils.slice_dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train UNetConcatControlPACA")

    # Required
    p.add_argument("--manifest",  required=True, help="manifest CSV produced by preprocess_synthrad_dataset.py")
    p.add_argument("--vae-path",  required=True, help="path to frozen VAE checkpoint (.pth)")

    # Output
    p.add_argument("--save-dir",  default="checkpoints/concat_paca",
                   help="directory for checkpoints and prediction samples")

    # Data
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--no-augment",  action="store_true",
                   help="disable training augmentation (flip + rotation)")

    # Model
    p.add_argument("--base-channels", type=int, default=64,
                   help="base channel width for UNet and ControlNet "
                        "(default 64 → ~31M trainable; 128 → ~124M; 256 → ~497M)")
    p.add_argument("--dropout-rate", type=float, default=0.0,
                   help="dropout rate for the UNet backbone")
    p.add_argument("--use-dr", action="store_true",
                   help="enable DR module; valid with --use-controlnet --control-source dr")
    p.add_argument("--use-controlnet", action="store_true",
                   help="enable ControlNet conditioning")
    p.add_argument("--control-source", choices=["dr", "cbct_latent"], default="cbct_latent",
                   help="ControlNet condition source")
    p.add_argument("--controlnet-fusion", choices=["add", "paca", "both"], default="add",
                   help="how ControlNet residuals enter UNet")
    p.add_argument("--latent-mode", choices=["mu", "sample"], default="mu",
                   help="VAE latent used by diffusion training")

    # Training
    p.add_argument("--epochs",         type=int,   default=300)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--warmup-steps",   type=int,   default=1000)
    p.add_argument("--use-ema", action="store_true",
                   help="enable UNet EMA for validation, decoded metrics, and saved unet_ema.pth")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--gamma",          type=float, default=1.0,
                   help="weight for DR auxiliary loss: total = L_diff + gamma * L_dr")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="number of micro-batches to accumulate before optimizer.step()")
    p.add_argument("--no-amp", action="store_true",
                   help="disable CUDA AMP autocast and GradScaler")
    p.add_argument("--early-stopping", type=int,   default=50,
                   help="stop if val loss doesn't improve for this many epochs")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="optional smoke-test limit for train batches per epoch")
    p.add_argument("--max-val-batches", type=int, default=None,
                   help="optional smoke-test limit for val batches per epoch")
    p.add_argument("--ddim-steps", type=int, default=50,
                   help="DDIM steps for decoded validation metrics and fixed-val images")
    p.add_argument("--sampler-init", choices=["noise", "cbct"], default="noise",
                   help="initial latent for decoded validation sampling")
    p.add_argument("--sampler-t-start", type=int, default=999,
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
    p.add_argument("--exp-id", type=str, default="manual",
                   help="experiment id used in WandB name/tags, e.g. A0/C1/B1")
    p.add_argument("--stage", type=str, default="smoke",
                   choices=["smoke", "screen", "long", "manual"])
    p.add_argument("--fixed-val-config", type=str, default="configs/fixed_val_cases.yaml")
    p.add_argument("--fixed-val-cases-per-region", type=int, default=4)
    p.add_argument("--fixed-val-slices-per-case", type=int, default=3)
    p.add_argument("--fixed-val-max-images", type=int, default=48,
                   help="number of fixed validation slices kept for WandB visualization")

    # Resume
    p.add_argument("--unet-path",       default=None,
                   help="resume: path to saved UNet state dict (unet_full.pth)")
    p.add_argument("--ema-path",        default=None,
                   help="resume: path to saved UNet EMA state dict (unet_ema_state.pth)")
    p.add_argument("--controlnet-path", default=None,
                   help="resume: path to saved ControlNet state dict")
    p.add_argument("--dr-path",         default=None,
                   help="resume: path to saved DegradationRemoval state dict")
    p.add_argument("--control-adapter-path", default=None,
                   help="resume: path to saved CBCT latent ControlNet adapter")

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
    print("UNetConcatControlPACA Training")
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
    # VAE — frozen, used only for encode/decode
    vae = load_vae(save_path=args.vae_path, trainable=False)
    vae.eval()
    print(f"VAE loaded (frozen): {args.vae_path}")

    # UNet
    unet = UNetConcatControlPACA(
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

    controlnet = ControlNet(in_channels=3, base_channels=args.base_channels) if args.use_controlnet else None
    if args.controlnet_path and controlnet is None:
        raise ValueError("--controlnet-path requires --use-controlnet")
    if args.controlnet_path:
        controlnet.load_state_dict(torch.load(args.controlnet_path, map_location="cpu"))
        print(f"ControlNet resumed from: {args.controlnet_path}")

    dr_module = DegradationRemoval(condition_channels=1, final_embedding_channels=args.base_channels) if args.use_dr else None
    if args.dr_path and dr_module is None:
        raise ValueError("--dr-path requires --use-dr")
    if args.dr_path:
        dr_module.load_state_dict(torch.load(args.dr_path, map_location="cpu"))
        print(f"DR module resumed from: {args.dr_path}")

    control_adapter = None
    if args.use_controlnet and args.control_source == "cbct_latent":
        control_adapter = LatentControlAdapter(in_channels=3, out_channels=args.base_channels)
        if args.control_adapter_path:
            control_adapter.load_state_dict(torch.load(args.control_adapter_path, map_location="cpu"))
            print(f"Control adapter resumed from: {args.control_adapter_path}")

    # Parameter summary
    n_unet = sum(p.numel() for p in unet.parameters())
    n_cn   = sum(p.numel() for p in controlnet.parameters()) if controlnet else 0
    n_dr   = sum(p.numel() for p in dr_module.parameters()) if dr_module else 0
    n_ad   = sum(p.numel() for p in control_adapter.parameters()) if control_adapter else 0
    print(f"\nTrainable parameters:")
    print(f"  UNet        : {n_unet/1e6:.1f}M")
    print(f"  ControlNet  : {n_cn/1e6:.1f}M")
    print(f"  DR module   : {n_dr/1e6:.3f}M")
    print(f"  Adapter     : {n_ad/1e6:.3f}M")
    print(f"  Total       : {(n_unet+n_cn+n_dr+n_ad)/1e6:.1f}M")

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
        f"Epochs: {args.epochs}  LR: {args.lr}  wd: {args.weight_decay}  gamma: {args.gamma}  "
        f"dropout: {args.dropout_rate}  "
        f"EMA: {'on' if args.use_ema else 'off'}  "
        f"grad-accum: {args.grad_accum_steps}  AMP: {'off' if args.no_amp else 'on'}  "
        f"early-stop: {args.early_stopping}\n"
        f"use_dr={args.use_dr} use_controlnet={args.use_controlnet} "
        f"control_source={args.control_source} fusion={args.controlnet_fusion}"
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
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "augmentation": augmentation,
                "base_channels": args.base_channels,
                "dropout_rate": args.dropout_rate,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
                "gamma": args.gamma,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch": args.batch_size * args.grad_accum_steps,
                "amp": not args.no_amp,
                "seed": args.seed,
                "use_dr": args.use_dr,
                "use_controlnet": args.use_controlnet,
                "control_source": args.control_source,
                "controlnet_fusion": args.controlnet_fusion,
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
            tags=["concat-paca", "phase1", args.exp_id, args.stage, commit],
            group=args.wandb_group,
            notes="UNetConcatControlPACA Phase 1 matrix training",
        )

    try:
        train_unet_concat_control_paca(
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            dr_module=dr_module,
            control_adapter=control_adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_dir=save_dir,
            predict_dir=predict_dir,
            early_stopping=args.early_stopping,
            gamma=args.gamma,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            ema_path=args.ema_path,
            wandb_logger=wandb_logger,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            grad_accum_steps=args.grad_accum_steps,
            amp_enabled=torch.cuda.is_available() and not args.no_amp,
            use_dr=args.use_dr,
            use_controlnet=args.use_controlnet,
            control_source=args.control_source,
            controlnet_fusion=args.controlnet_fusion,
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
