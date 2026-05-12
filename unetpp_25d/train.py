import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.unetConcatControlPACA import _make_comparison_panel, _to_image01
from unetpp_25d.dataset import get_dataloaders
from unetpp_25d.eval import evaluate_model
from unetpp_25d.losses import total_loss
from unetpp_25d.model import UNetPlusPlus25D
from utils.hu import to_hu


def parse_args():
    p = argparse.ArgumentParser(description="Train 2.5D U-Net++ for CBCT-to-sCT")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--save-dir", default="checkpoints/unetpp_25d/UNetPP25D-resnet34-s7-bs32-l1grad-s42")
    p.add_argument("--input-slices", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--mask-min-pixels", type=int, default=100)

    p.add_argument("--encoder-name", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--no-clamp-output", action="store_true")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-weight", type=float, default=0.05)
    p.add_argument("--full-mae-weight", type=float, default=0.0)
    p.add_argument("--ms-ssim-weight", type=float, default=0.0)
    p.add_argument("--high-density-weight", type=float, default=0.0)
    p.add_argument("--high-density-threshold-hu", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--max-eval-patients", type=int, default=None)
    p.add_argument("--fixed-val-max-images", type=int, default=16)

    p.add_argument("--resume", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="cbct2sct_IBA")
    p.add_argument("--wandb-group", default="unetpp25d-v2")
    p.add_argument("--wandb-name", default="UNetPP25D-resnet34-s7-bs32-l1grad-s42")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_dtype(name: str):
    return torch.float16 if name == "fp16" else torch.bfloat16


def run_epoch(model, loader, optimizer, scaler, device, args, train: bool):
    model.train(train)
    total = l1_total = grad_total = full_mae_total = ms_ssim_total = high_density_total = 0.0
    n = 0
    max_batches = args.max_train_batches if train else args.max_val_batches
    amp_enabled = device.type == "cuda" and not args.no_amp
    dtype = autocast_dtype(args.amp_dtype)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        cbct = batch["cbct"].to(device, non_blocking=True)
        ct = batch["ct"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=amp_enabled, dtype=dtype):
                pred = model(cbct)
                loss, parts = total_loss(
                    pred, ct, mask,
                    grad_weight=args.grad_weight,
                    full_mae_weight=args.full_mae_weight,
                    ms_ssim_weight=args.ms_ssim_weight,
                    high_density_weight=args.high_density_weight,
                    high_density_threshold_hu=args.high_density_threshold_hu,
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total += float(loss.detach())
        l1_total += parts["loss/l1"]
        grad_total += parts["loss/grad"]
        full_mae_total += parts["loss/full_mae"]
        ms_ssim_total += parts["loss/ms_ssim"]
        high_density_total += parts["loss/high_density"]
        n += 1

    denom = max(n, 1)
    return {
        "loss": total / denom,
        "l1": l1_total / denom,
        "grad": grad_total / denom,
        "full_mae": full_mae_total / denom,
        "ms_ssim": ms_ssim_total / denom,
        "high_density": high_density_total / denom,
        "batches": n,
    }


@torch.no_grad()
def log_fixed_val(model, val_dataset, device, wandb_logger, epoch, max_images, args):
    if wandb_logger is None or max_images <= 0:
        return
    items = val_dataset.fixed_val_items(cases_per_region=4, slices_per_case=3)[:max_images]
    if not items:
        return
    model.eval()
    amp_enabled = device.type == "cuda" and not args.no_amp
    dtype = autocast_dtype(args.amp_dtype)
    for idx, item in enumerate(items):
        cbct = item["cbct"].unsqueeze(0).to(device)
        with autocast(enabled=amp_enabled, dtype=dtype):
            pred = model(cbct)
        ct = item["ct"].unsqueeze(0)
        mask = item["mask"].unsqueeze(0)
        center = args.input_slices // 2
        cbct_center = cbct.detach().float().cpu()[0, center]
        sct = pred.detach().float().cpu()[0, 0]
        ct0 = ct[0, 0]
        mask0 = mask[0, 0]
        err_hu = (to_hu(sct) - to_hu(ct0)).abs() * mask0
        mae_hu = float(err_hu.sum() / mask0.sum().clamp_min(1.0))
        panel = _make_comparison_panel(
            _to_image01(cbct_center).numpy(),
            _to_image01(ct0).numpy(),
            _to_image01(sct).numpy(),
            (err_hu.clamp(0, 300) / 300.0).numpy(),
            error_max_hu=300,
            error_label=f"|err| MAE {mae_hu:.1f}HU",
        )
        caption = f"{item['patient_id']} {item['region']} z{item['z']} | MAE {mae_hu:.1f} HU"
        wandb_logger.log_image(f"fixed_val/sample_{idx:02d}", panel, caption=caption, step=epoch)


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(device_name)
    save_dir = os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("UNet++ 2.5D Training")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device  : {device}")
    print("=" * 60)

    train_loader, val_loader = get_dataloaders(
        args.manifest,
        batch_size=args.batch_size,
        input_slices=args.input_slices,
        num_workers=args.num_workers,
        augmentation=not args.no_augment,
        seed=args.seed,
        mask_min_pixels=args.mask_min_pixels,
    )

    model = UNetPlusPlus25D(
        in_channels=args.input_slices,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        clamp_output=not args.no_clamp_output,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if device.type == "cuda" and not args.no_amp and args.amp_dtype == "fp16" else None
    start_epoch = 0
    best_loss = float("inf")
    best_mae = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")
    print(
        f"Run: {args.wandb_name}\n"
        f"save_dir={save_dir}\n"
        f"input_slices={args.input_slices} batch={args.batch_size} lr={args.lr} "
        f"grad_weight={args.grad_weight} full_mae_weight={args.full_mae_weight} "
        f"ms_ssim_weight={args.ms_ssim_weight} high_density_weight={args.high_density_weight} "
        f"high_density_threshold_hu={args.high_density_threshold_hu} "
        f"amp={'off' if args.no_amp else args.amp_dtype}"
    )

    wandb_logger = None
    if not args.no_wandb:
        from utils.wandb_logger import WandbLogger
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group,
            config=vars(args),
            tags=["unetpp25d", "resnet34", "v2", f"s{args.input_slices}"],
            notes="2.5D U-Net++ direct CT regression baseline.",
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_stats = run_epoch(model, train_loader, optimizer, scaler, device, args, train=True)
            with torch.no_grad():
                val_stats = run_epoch(model, val_loader, optimizer, scaler, device, args, train=False)
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            ep = epoch + 1
            print(
                f"Epoch {ep} | Train {train_stats['loss']:.6f} "
                f"(L1 {train_stats['l1']:.6f}, Grad {train_stats['grad']:.6f}, "
                f"Full {train_stats['full_mae']:.6f}, MS {train_stats['ms_ssim']:.6f}, "
                f"High {train_stats['high_density']:.6f}) | "
                f"Val {val_stats['loss']:.6f} "
                f"(L1 {val_stats['l1']:.6f}, Grad {val_stats['grad']:.6f}, "
                f"Full {val_stats['full_mae']:.6f}, MS {val_stats['ms_ssim']:.6f}, "
                f"High {val_stats['high_density']:.6f}) | "
                f"LR {lr:.2e} | {elapsed:.1f}s | GPU {gpu_mem:.2f}GB"
            )

            metrics = {
                "train/l1": train_stats["l1"],
                "train/grad": train_stats["grad"],
                "train/full_mae": train_stats["full_mae"],
                "train/ms_ssim_loss": train_stats["ms_ssim"],
                "train/high_density_l1": train_stats["high_density"],
                "train/batches": train_stats["batches"],
                "val/l1": val_stats["l1"],
                "val/grad": val_stats["grad"],
                "val/full_mae": val_stats["full_mae"],
                "val/ms_ssim_loss": val_stats["ms_ssim"],
                "val/high_density_l1": val_stats["high_density"],
                "val/batches": val_stats["batches"],
                "gpu_mem_max_gb": gpu_mem,
                "epoch_time_sec": elapsed,
            }

            if ep % max(args.eval_every, 1) == 0:
                eval_metrics = evaluate_model(
                    model, val_loader.dataset, device,
                    batch_size=args.batch_size,
                    amp_enabled=device.type == "cuda" and not args.no_amp,
                    amp_dtype=args.amp_dtype,
                    max_patients=args.max_eval_patients,
                )
                metrics.update(eval_metrics)
                log_fixed_val(model, val_loader.dataset, device, wandb_logger, ep, args.fixed_val_max_images, args)
                mae = eval_metrics.get("val/mae_hu", float("inf"))
                if mae < best_mae:
                    best_mae = float(mae)
                    save_checkpoint(os.path.join(save_dir, "model_best_mae.pth"), model, optimizer, ep, args, metrics)
                    print(f"Saved best MAE epoch {ep}: {best_mae:.2f} HU")

            save_checkpoint(os.path.join(save_dir, "model_latest.pth"), model, optimizer, ep, args, metrics)
            if val_stats["loss"] < best_loss:
                best_loss = float(val_stats["loss"])
                save_checkpoint(os.path.join(save_dir, "model_best_loss.pth"), model, optimizer, ep, args, metrics)
                print(f"Saved best loss epoch {ep}: {best_loss:.6f}")

            if wandb_logger:
                wandb_logger.log_training_step(
                    epoch=ep,
                    train_loss=train_stats["loss"],
                    val_loss=val_stats["loss"],
                    learning_rate=lr,
                    extra_metrics=metrics,
                )
    finally:
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
