#!/usr/bin/env python
"""Train a legacy-inspired Concatenation + CFG protocol on current data.

This entry point intentionally uses the old source-of-truth model
``models.unetConditional.UNetConcatenation`` instead of the newer PACA/ControlNet
training wrapper. The goal is not bitwise reproduction of the old NPY pipeline;
it keeps the legacy CFG concat design while using current-data improvements:
SliceDataset, mask-weighted latent loss by default, W&B loss curves, fixed-val
panels, and mask-based decoded metrics.

Default recipe follows the old successful direction with current pipeline
optimizations:
  - legacy concat UNet, base_channels=256, dropout=0.1
  - CFG condition dropout enabled, cfg_dropout_rate=0.15
  - BF16 AMP, AdamW lr=1e-4, WSD schedule, EMA decay=0.9999
  - deterministic VAE mu latent by default (sample remains available for legacy ablation)
  - mask-weighted latent diffusion loss by default
  - pure-noise DDIM40 validation sampler by default
"""
import argparse
import gc
import os
import random
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    ExponentialMovingAverage = None
    HAS_EMA = False

from models.diffusion import Diffusion
from models.unetConditional import UNetConcatenation
from models.unetConcatControlPACA import (
    _make_comparison_panel,
    _masked_psnr,
    _masked_ssim,
    _to_hu,
    _to_image01,
)
from models.vae import load_vae
from utils.slice_dataset import get_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train legacy UNetConcatenation + CFG on current manifest")

    p.add_argument("--manifest", required=True, help="current preprocessed manifest CSV")
    p.add_argument("--vae-path", required=True, help="frozen VAE checkpoint")
    p.add_argument("--save-dir", default="checkpoints/phase1_matrix/L0-cfg-legacy")

    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--base-channels", type=int, default=256)
    p.add_argument("--dropout-rate", type=float, default=0.1)
    p.add_argument("--latent-mode", choices=["mu", "sample"], default="mu",
                   help="mu is deterministic and train/eval consistent; sample matches old train_cfg.py")
    p.add_argument("--loss-scope", choices=["mask", "full"], default="mask",
                   help="mask is current optimized default; full matches old train_cfg.py diffusion loss")

    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--lr-schedule", choices=["wsd", "cosine", "constant"], default="wsd")
    p.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="old WSD/cosine warmup ratio")
    p.add_argument("--stable-ratio", type=float, default=0.70,
                   help="old WSD stable phase end ratio")
    p.add_argument("--early-stopping", type=int, default=40)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)

    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--no-ema", action="store_true", help="disable EMA; EMA is on by default")
    p.add_argument("--ema-decay", type=float, default=0.9999)

    p.add_argument("--no-cfg", action="store_true", help="disable CFG condition dropout")
    p.add_argument("--cfg-dropout-rate", type=float, default=0.15)
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="decoded validation guidance scale; 1.0 is conditional DDIM")

    p.add_argument("--ddim-steps", type=int, default=40)
    p.add_argument("--sampler-init", choices=["noise", "cbct"], default="noise")
    p.add_argument("--sampler-t-start", type=int, default=999)
    p.add_argument("--sampler-alpha", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=10)

    p.add_argument("--fixed-val-config", type=str, default="configs/fixed_val_cases.yaml")
    p.add_argument("--fixed-val-cases-per-region", type=int, default=4)
    p.add_argument("--fixed-val-slices-per-case", type=int, default=3)
    p.add_argument("--fixed-val-max-images", type=int, default=48)

    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="cbct2sct_IBA")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default="phase1-matrix-2026-05")
    p.add_argument("--exp-id", type=str, default="L0-CFG")
    p.add_argument("--stage", choices=["smoke", "screen", "strong", "long", "manual"], default="strong")

    p.add_argument("--unet-path", default=None, help="resume/load raw or EMA UNet state dict")
    p.add_argument("--checkpoint-path", default=None, help="resume full training checkpoint")

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_short_sha():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
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
    return (
        torch.stack([x["ct"] for x in items]),
        torch.stack([x["cbct"] for x in items]),
        torch.stack([x["mask"] for x in items]),
        [f"{x['patient_id']} {x['region']} z{x['z']}" for x in items],
    )


def autocast_dtype(precision):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def encode_vae(vae, x, latent_mode):
    mu, logvar = vae.encode(x)
    if latent_mode == "sample":
        return vae.reparameterize(mu, logvar)
    return mu


def latent_mask_from_pixel_mask(mask, latent_hw):
    stride_h = max(mask.shape[-2] // latent_hw[0], 1)
    stride_w = max(mask.shape[-1] // latent_hw[1], 1)
    pooled = F.avg_pool2d(mask.float(), kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w))
    if pooled.shape[-2:] != latent_hw:
        pooled = F.interpolate(pooled, size=latent_hw, mode="nearest")
    return pooled > 0.5


def masked_mse(pred, target, mask):
    mask = mask.to(dtype=pred.dtype)
    denom = (mask.sum() * pred.size(1)).clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def build_scheduler(optimizer, schedule, total_steps, warmup_ratio, stable_ratio):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(total_steps * warmup_ratio), 1)
    stable_steps = max(int(total_steps * stable_ratio), warmup_steps + 1)

    if schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0), warmup_steps, stable_steps

    if schedule == "wsd":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            if step < stable_steps:
                return 1.0
            progress = float(step - stable_steps) / float(max(total_steps - stable_steps, 1))
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warmup_steps, stable_steps

    def cosine_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda), warmup_steps, stable_steps


def predict_noise(unet, z, cbct_z, t, cfg_scale):
    if cfg_scale <= 1.0:
        return unet(z, cbct_z, t)
    eps_cond = unet(z, cbct_z, t)
    eps_uncond = unet(z, torch.zeros_like(cbct_z), t)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def ddim_sample(vae, unet, diffusion, cbct_img, latent_mode, steps, precision, sampler_init, sampler_t_start,
                sampler_alpha, cfg_scale):
    amp_enabled = torch.cuda.is_available() and precision != "fp32"
    cbct_z = encode_vae(vae, cbct_img, latent_mode)
    sampler_t_start = int(min(max(sampler_t_start, 0), diffusion.timesteps - 1))
    timesteps = torch.linspace(sampler_t_start, 0, steps, device=cbct_z.device).long()
    generator = torch.Generator(device=cbct_z.device).manual_seed(0)
    noise = torch.randn(cbct_z.shape, device=cbct_z.device, dtype=cbct_z.dtype, generator=generator)

    if sampler_init == "noise":
        z = noise
    else:
        alpha_eff = (sampler_alpha * diffusion.alpha_cumprod[timesteps[0]]).clamp(0.0, 1.0).view(1, 1, 1, 1)
        z = torch.sqrt(alpha_eff) * cbct_z + torch.sqrt(1.0 - alpha_eff) * noise

    for idx in range(len(timesteps) - 1):
        t_scalar = timesteps[idx]
        t_prev = timesteps[idx + 1]
        t = torch.full((cbct_z.size(0),), int(t_scalar.item()), device=cbct_z.device, dtype=torch.long)
        with autocast(enabled=amp_enabled, dtype=autocast_dtype(precision)):
            eps = predict_noise(unet, z, cbct_z, t, cfg_scale)
        alpha_t = diffusion.alpha_cumprod[t_scalar].view(1, 1, 1, 1)
        pred_x0 = (z - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
        alpha_prev = diffusion.alpha_cumprod[t_prev].view(1, 1, 1, 1)
        z = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * eps

    return vae.decode(z).clamp(-1.0, 1.0)


@torch.no_grad()
def eval_decoded(vae, unet, diffusion, val_loader, device, args):
    totals = {"mae_hu": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    for i, (ct_img, cbct_img, mask, _) in enumerate(val_loader):
        if args.max_val_batches is not None and i >= args.max_val_batches:
            break
        ct_img = ct_img.to(device)
        cbct_img = cbct_img.to(device)
        mask = mask.to(device)
        sct = ddim_sample(
            vae, unet, diffusion, cbct_img, args.latent_mode, args.ddim_steps, args.precision,
            args.sampler_init, args.sampler_t_start, args.sampler_alpha, args.cfg_scale,
        )
        sct = sct * mask + (-1.0) * (1.0 - mask)
        mae_per = ((_to_hu(sct) - _to_hu(ct_img)).abs() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0)
        batch_n = ct_img.size(0)
        totals["mae_hu"] += float(mae_per.mean()) * batch_n
        totals["psnr"] += _masked_psnr(sct, ct_img, mask) * batch_n
        totals["ssim"] += _masked_ssim(sct, ct_img, mask) * batch_n
        count += batch_n
    return {f"val/{k}": v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def log_fixed_val_images(vae, unet, diffusion, fixed_val_batch, device, args, wandb_logger, epoch):
    if wandb_logger is None or fixed_val_batch is None:
        return
    ct_img, cbct_img, mask, captions = fixed_val_batch
    total = min(args.fixed_val_max_images, ct_img.size(0))
    chunk = min(8, max(total, 1))
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        ct = ct_img[start:end].to(device)
        cbct = cbct_img[start:end].to(device)
        m = mask[start:end].to(device)
        sct = ddim_sample(
            vae, unet, diffusion, cbct, args.latent_mode, args.ddim_steps, args.precision,
            args.sampler_init, args.sampler_t_start, args.sampler_alpha, args.cfg_scale,
        )
        sct = sct * m + (-1.0) * (1.0 - m)
        err = ((_to_hu(sct) - _to_hu(ct)).abs() * m).clamp(0, 300) / 300.0
        for local_i in range(ct.size(0)):
            idx = start + local_i
            panel = _make_comparison_panel(
                _to_image01(cbct[local_i]).squeeze().detach().cpu().numpy(),
                _to_image01(ct[local_i]).squeeze().detach().cpu().numpy(),
                _to_image01(sct[local_i]).squeeze().detach().cpu().numpy(),
                err[local_i].squeeze().detach().cpu().numpy(),
            )
            wandb_logger.log_image(f"fixed_val_legacy/comparison_{idx}", panel, captions[idx], step=epoch)


def cpu_state_dict(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def save_best(save_dir, unet, ema, epoch, val_loss):
    raw_path = os.path.join(save_dir, "unet_full.pth")
    best_path = os.path.join(save_dir, "unet_best.pth")
    torch.save(cpu_state_dict(unet), raw_path)
    if ema is not None:
        with ema.average_parameters():
            ema_state = cpu_state_dict(unet)
        torch.save(ema_state, os.path.join(save_dir, "unet_ema.pth"))
        torch.save(ema_state, best_path)
        torch.save(ema.state_dict(), os.path.join(save_dir, "unet_ema_state.pth"))
    else:
        torch.save(cpu_state_dict(unet), best_path)
    print(f"Saved best epoch {epoch}: val_loss={val_loss:.6f} -> {best_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = git_short_sha()
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    use_cfg = not args.no_cfg
    use_ema = not args.no_ema
    if use_ema and not HAS_EMA:
        raise RuntimeError("torch-ema is required for default EMA; rerun with --no-ema to disable it")
    if args.precision == "bf16" and torch.cuda.is_available():
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not bf16_supported:
            raise RuntimeError("BF16 requested but torch.cuda.is_bf16_supported() is false")

    print("=" * 72)
    print("Legacy UNetConcatenation + CFG Training")
    print(f"Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device       : {device}")
    print(f"Commit       : {commit}")
    print(f"Save dir     : {save_dir}")
    print("=" * 72)

    train_loader, val_loader = get_dataloaders(
        manifest_csv=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=not args.no_augment,
        seed=args.seed,
    )
    fixed_val_batch = build_fixed_val_batch(
        val_loader.dataset,
        args.fixed_val_config,
        args.fixed_val_cases_per_region,
        args.fixed_val_slices_per_case,
        args.fixed_val_max_images,
    )

    vae = load_vae(args.vae_path, trainable=False).to(device).eval()
    unet = UNetConcatenation(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        dropout_rate=args.dropout_rate,
    ).to(device)
    if args.unet_path:
        unet.load_state_dict(torch.load(args.unet_path, map_location="cpu"), strict=True)
        print(f"Loaded UNet state: {args.unet_path}")

    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    train_batches = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
    optim_steps_per_epoch = (train_batches + args.grad_accum_steps - 1) // args.grad_accum_steps
    total_steps = max(args.epochs * optim_steps_per_epoch, 1)
    scheduler, warmup_steps, stable_steps = build_scheduler(
        optimizer, args.lr_schedule, total_steps, args.warmup_ratio, args.stable_ratio
    )
    diffusion = Diffusion(device)

    amp_enabled = torch.cuda.is_available() and args.precision != "fp32"
    scaler = GradScaler(enabled=amp_enabled and args.precision == "fp16")
    ema = ExponentialMovingAverage(unet.parameters(), decay=args.ema_decay) if use_ema else None
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        unet.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        global_step = int(ckpt.get("global_step", 0))
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"Resumed checkpoint {args.checkpoint_path} at epoch {start_epoch}")

    print(f"Train slices : {len(train_loader.dataset)}")
    print(f"Val slices   : {len(val_loader.dataset)}")
    print(f"Model params : {sum(p.numel() for p in params) / 1e6:.2f}M")
    print(f"CFG          : {use_cfg} dropout={args.cfg_dropout_rate} eval_scale={args.cfg_scale}")
    print(f"Latent mode  : {args.latent_mode}")
    print(f"Loss scope   : {args.loss_scope}")
    print(f"Precision    : {args.precision} amp={amp_enabled}")
    print(f"EMA          : {use_ema} decay={args.ema_decay}")
    print(
        f"Schedule     : {args.lr_schedule} total_steps={total_steps} "
        f"warmup={warmup_steps} stable={stable_steps}"
    )
    print(f"Sampler      : {args.sampler_init} t_start={args.sampler_t_start} alpha={args.sampler_alpha} steps={args.ddim_steps}")

    wandb_logger = None
    if not args.no_wandb:
        from utils.wandb_logger import WandbLogger
        run_name = args.wandb_name or f"{args.exp_id}-legacy-cfg-bc{args.base_channels}-ep{args.epochs}-s{args.seed}"
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            config={
                "exp_id": args.exp_id,
                "stage": args.stage,
                "commit": commit,
                "script": "scripts/train_legacy_cfg.py",
                "model": "models.unetConditional.UNetConcatenation",
                "manifest": args.manifest,
                "vae_path": args.vae_path,
                "save_dir": args.save_dir,
                "train_slices": len(train_loader.dataset),
                "val_slices": len(val_loader.dataset),
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch": args.batch_size * args.grad_accum_steps,
                "base_channels": args.base_channels,
                "dropout_rate": args.dropout_rate,
                "latent_mode": args.latent_mode,
                "loss_scope": args.loss_scope,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "gradient_clip": args.gradient_clip,
                "lr_schedule": args.lr_schedule,
                "warmup_ratio": args.warmup_ratio,
                "stable_ratio": args.stable_ratio,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "stable_steps": stable_steps,
                "precision": args.precision,
                "use_ema": use_ema,
                "ema_decay": args.ema_decay,
                "use_cfg": use_cfg,
                "cfg_dropout_rate": args.cfg_dropout_rate if use_cfg else 0.0,
                "cfg_scale": args.cfg_scale,
                "ddim_steps": args.ddim_steps,
                "sampler_init": args.sampler_init,
                "sampler_t_start": args.sampler_t_start,
                "sampler_alpha": args.sampler_alpha,
                "region_embedding": False,
                "use_controlnet": False,
                "use_dr": False,
                "seed": args.seed,
            },
            tags=["legacy-cfg", "concat", args.exp_id, args.stage, commit],
            notes="Migrated legacy Concatenation+CFG recipe onto current data/metrics; old legacy files unchanged.",
        )

    early_counter = 0
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            unet.train()
            train_loss_total = 0.0
            valid_batches = 0
            skipped_nonfinite = 0
            skipped_overflow = 0
            optimizer_steps = 0
            accum_batches = 0
            optimizer.zero_grad(set_to_none=True)

            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for i, (ct_img, cbct_img, mask, _) in enumerate(progress):
                if args.max_train_batches is not None and i >= args.max_train_batches:
                    break
                ct_img = ct_img.to(device)
                cbct_img = cbct_img.to(device)
                mask = mask.to(device)

                with torch.no_grad(), autocast(enabled=amp_enabled, dtype=autocast_dtype(args.precision)):
                    ct_z = encode_vae(vae, ct_img, args.latent_mode)
                    cbct_z = encode_vae(vae, cbct_img, args.latent_mode)

                with autocast(enabled=amp_enabled, dtype=autocast_dtype(args.precision)):
                    t = diffusion.sample_timesteps(ct_z.size(0))
                    noise = torch.randn_like(ct_z)
                    z_noisy = diffusion.add_noise(ct_z, t, noise=noise)
                    if use_cfg:
                        keep = (torch.rand(cbct_z.size(0), 1, 1, 1, device=device) >= args.cfg_dropout_rate).to(cbct_z.dtype)
                        cbct_in = cbct_z * keep
                    else:
                        cbct_in = cbct_z
                    pred_noise = unet(z_noisy, cbct_in, t)
                    if args.loss_scope == "mask":
                        latent_mask = latent_mask_from_pixel_mask(mask, pred_noise.shape[-2:])
                        loss = masked_mse(pred_noise, noise, latent_mask)
                    else:
                        loss = F.mse_loss(pred_noise, noise)
                    scaled_loss = loss / max(args.grad_accum_steps, 1)

                if not torch.isfinite(loss.detach()):
                    skipped_nonfinite += 1
                    accum_batches = 0
                    optimizer.zero_grad(set_to_none=True)
                    print(f"Warning: non-finite loss at epoch {epoch + 1}, batch {i + 1}; skipped")
                    continue

                scaler.scale(scaled_loss).backward()
                accum_batches += 1
                should_step = accum_batches >= args.grad_accum_steps or (i + 1) == train_batches
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=args.gradient_clip)
                    scale_before = scaler.get_scale()
                    if not torch.isfinite(grad_norm):
                        skipped_overflow += 1
                        optimizer.zero_grad(set_to_none=True)
                        accum_batches = 0
                        print(f"Warning: non-finite grad_norm at epoch {epoch + 1}, batch {i + 1}; skipped bookkeeping")
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scale_after = scaler.get_scale()
                    overflow = scaler.is_enabled() and scale_after < scale_before
                    if overflow:
                        skipped_overflow += 1
                    else:
                        scheduler.step()
                        if ema is not None:
                            ema.update()
                        optimizer_steps += 1
                        global_step += 1
                    accum_batches = 0

                loss_value = float(loss.detach())
                train_loss_total += loss_value
                valid_batches += 1
                progress.set_postfix({"loss": f"{loss_value:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            train_loss = train_loss_total / max(valid_batches, 1)

            if ema is not None:
                ema.store()
                ema.copy_to()
            save_this_epoch = False
            best_epoch_loss = None
            try:
                unet.eval()
                val_loss_total = 0.0
                val_batches = 0
                val_generator = torch.Generator(device=device).manual_seed(42)
                with torch.no_grad():
                    for i, (ct_img, cbct_img, mask, _) in enumerate(val_loader):
                        if args.max_val_batches is not None and i >= args.max_val_batches:
                            break
                        ct_img = ct_img.to(device)
                        cbct_img = cbct_img.to(device)
                        mask = mask.to(device)
                        with autocast(enabled=amp_enabled, dtype=autocast_dtype(args.precision)):
                            ct_z = encode_vae(vae, ct_img, args.latent_mode)
                            cbct_z = encode_vae(vae, cbct_img, args.latent_mode)
                            t = diffusion.sample_timesteps(ct_z.size(0), generator=val_generator)
                            noise = torch.randn(ct_z.shape, device=device, dtype=ct_z.dtype, generator=val_generator)
                            z_noisy = diffusion.add_noise(ct_z, t, noise=noise)
                            pred_noise = unet(z_noisy, cbct_z, t)
                            if args.loss_scope == "mask":
                                latent_mask = latent_mask_from_pixel_mask(mask, pred_noise.shape[-2:])
                                val_loss = masked_mse(pred_noise, noise, latent_mask)
                            else:
                                val_loss = F.mse_loss(pred_noise, noise)
                        val_loss_total += float(val_loss.detach())
                        val_batches += 1

                val_loss = val_loss_total / max(val_batches, 1)
                epoch_time = time.time() - epoch_start
                gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                current_lr = optimizer.param_groups[0]["lr"]
                step_time_ms = epoch_time * 1000.0 / max(train_batches, 1)

                extra = {
                    "train/loss_diff": train_loss,
                    "val/loss_diff": val_loss,
                    "train/loss_dr": 0.0,
                    "val/loss_dr": 0.0,
                    "train/valid_batches": valid_batches,
                    "train/skipped_nonfinite_batches": skipped_nonfinite,
                    "train/skipped_overflow_steps": skipped_overflow,
                    "train/optimizer_steps": optimizer_steps,
                    "train/global_step": global_step,
                    "lr/current": current_lr,
                    "lr/total_steps": total_steps,
                    "lr/warmup_steps": warmup_steps,
                    "gpu_mem_max_gb": gpu_mem,
                    "epoch_time_sec": epoch_time,
                    "step_time_ms": step_time_ms,
                    "trainable_params": sum(p.numel() for p in params),
                }
                if (epoch + 1) % max(args.eval_every, 1) == 0:
                    extra.update(eval_decoded(vae, unet, diffusion, val_loader, device, args))
                    log_fixed_val_images(vae, unet, diffusion, fixed_val_batch, device, args, wandb_logger, epoch + 1)

                if wandb_logger:
                    wandb_logger.log_training_step(
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        learning_rate=current_lr,
                        extra_metrics=extra,
                    )

                val_label = "ValEMA" if ema is not None else "Val"
                print(
                    f"Epoch {epoch + 1} | Train {train_loss:.6f} | {val_label} {val_loss:.6f} | "
                    f"LR {current_lr:.2e} | {epoch_time:.1f}s | GPU {gpu_mem:.2f}GB"
                )

                finite_epoch = np.isfinite(train_loss) and np.isfinite(val_loss)
                if finite_epoch and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_counter = 0
                    save_this_epoch = True
                    best_epoch_loss = val_loss
                else:
                    early_counter += 1
            finally:
                if ema is not None:
                    ema.restore()

            if save_this_epoch:
                save_best(save_dir, unet, ema, epoch + 1, best_epoch_loss)
                if wandb_logger:
                    wandb_logger.log_metrics({"best_epoch": epoch + 1, "best_val_loss": best_val_loss}, step=epoch + 1)

            if (epoch + 1) % 10 == 0:
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": cpu_state_dict(unet),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "global_step": global_step,
                }
                if scaler.is_enabled():
                    ckpt["scaler_state_dict"] = scaler.state_dict()
                if ema is not None:
                    ckpt["ema_state_dict"] = ema.state_dict()
                torch.save(ckpt, os.path.join(save_dir, "unet_last_checkpoint.pth"))

            if args.early_stopping and early_counter >= args.early_stopping:
                print(f"Early stopping after {args.early_stopping} epochs without val improvement")
                break
    finally:
        if wandb_logger:
            wandb_logger.finish()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 72)
    print(f"Training finished. Best val_loss={best_val_loss:.6f}")
    print(f"Checkpoints: {save_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
