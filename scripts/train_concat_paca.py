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
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models.vae import load_vae
from models.controlnet import ControlNet
from models.degradationRemoval import DegradationRemoval
from models.unetConcatControlPACA import UNetConcatControlPACA, train_unet_concat_control_paca
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
    p.add_argument("--base-channels", type=int, default=256,
                   help="base channel width for UNet and ControlNet (default 256 → 613M params)")

    # Training
    p.add_argument("--epochs",         type=int,   default=300)
    p.add_argument("--lr",             type=float, default=5e-6)
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

    # WandB
    p.add_argument("--no-wandb", action="store_true", help="disable WandB logging")
    p.add_argument("--wandb-project", type=str, default="cbct2sct_IBA",
                   help="WandB project name")
    p.add_argument("--wandb-name", type=str, default=None,
                   help="WandB run name")

    # Resume
    p.add_argument("--unet-path",       default=None,
                   help="resume: path to saved UNet state dict (unet_full.pth)")
    p.add_argument("--controlnet-path", default=None,
                   help="resume: path to saved ControlNet state dict")
    p.add_argument("--dr-path",         default=None,
                   help="resume: path to saved DegradationRemoval state dict")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    )
    if args.unet_path:
        state = torch.load(args.unet_path, map_location="cpu")
        missing, unexpected = unet.load_state_dict(state, strict=False)
        if missing:
            print(f"  UNet missing keys  : {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  UNet unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        print(f"UNet resumed from: {args.unet_path}")

    # ControlNet — encoder-only copy of UNet architecture
    controlnet = ControlNet(in_channels=3, base_channels=args.base_channels)
    if args.controlnet_path:
        controlnet.load_state_dict(torch.load(args.controlnet_path, map_location="cpu"))
        print(f"ControlNet resumed from: {args.controlnet_path}")

    # DegradationRemoval — lightweight pixel-space feature extractor (~0.3M)
    dr_module = DegradationRemoval(condition_channels=1, final_embedding_channels=args.base_channels)
    if args.dr_path:
        dr_module.load_state_dict(torch.load(args.dr_path, map_location="cpu"))
        print(f"DR module resumed from: {args.dr_path}")

    # Parameter summary
    n_unet = sum(p.numel() for p in unet.parameters())
    n_cn   = sum(p.numel() for p in controlnet.parameters())
    n_dr   = sum(p.numel() for p in dr_module.parameters())
    print(f"\nTrainable parameters:")
    print(f"  UNet        : {n_unet/1e6:.1f}M")
    print(f"  ControlNet  : {n_cn/1e6:.1f}M")
    print(f"  DR module   : {n_dr/1e6:.3f}M")
    print(f"  Total       : {(n_unet+n_cn+n_dr)/1e6:.1f}M")

    # ---- Train --------------------------------------------------------------
    save_dir    = os.path.join(PROJECT_ROOT, args.save_dir)
    predict_dir = os.path.join(save_dir, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)

    print(f"\nCheckpoints → {save_dir}")
    print(
        f"Epochs: {args.epochs}  LR: {args.lr}  gamma: {args.gamma}  "
        f"grad-accum: {args.grad_accum_steps}  AMP: {'off' if args.no_amp else 'on'}  "
        f"early-stop: {args.early_stopping}"
    )
    print("=" * 60 + "\n")

    wandb_logger = None
    if not args.no_wandb:
        from utils.wandb_logger import WandbLogger
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "manifest": args.manifest,
                "vae_path": args.vae_path,
                "save_dir": args.save_dir,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "augmentation": augmentation,
                "base_channels": args.base_channels,
                "epochs": args.epochs,
                "lr": args.lr,
                "gamma": args.gamma,
                "grad_accum_steps": args.grad_accum_steps,
                "amp": not args.no_amp,
                "early_stopping": args.early_stopping,
                "max_train_batches": args.max_train_batches,
                "max_val_batches": args.max_val_batches,
                "train_slices": len(train_loader.dataset),
                "val_slices": len(val_loader.dataset),
            },
            tags=["concat-paca", "phase1", "preprocessed-mha"],
            notes="UNetConcatControlPACA Phase 1 training",
        )

    try:
        train_unet_concat_control_paca(
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            dr_module=dr_module,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            save_dir=save_dir,
            predict_dir=predict_dir,
            early_stopping=args.early_stopping,
            gamma=args.gamma,
            learning_rate=args.lr,
            wandb_logger=wandb_logger,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            grad_accum_steps=args.grad_accum_steps,
            amp_enabled=torch.cuda.is_available() and not args.no_amp,
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
