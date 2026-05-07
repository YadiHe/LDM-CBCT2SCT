"""
VAE 重建基线评估：GT CT encode(mu) → decode → 计算 MAE/PSNR/SSIM。

用途：确认 VAE 重建本身的指标上界。若 mae_hu_vae ≈ D1 pilot (101 HU)，
说明瓶颈在 VAE，继续训练扩散模型收益有限；若 mae_hu_vae ≪ 101，继续 D1。

运行：
    python scripts/eval_vae_baseline.py \
        --manifest data/manifest.csv \
        --vae-path checkpoints/vae/vae_best.pth
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.vae import load_vae
from utils.slice_dataset import get_dataloaders
from models.unetConcatControlPACA import (
    _to_hu, _masked_psnr, _masked_ssim, REGION_NAMES
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",  required=True)
    p.add_argument("--vae-path",  required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae = load_vae(args.vae_path, trainable=False).to(device)

    _, val_loader = get_dataloaders(
        manifest_csv=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation=False,
    )
    print(f"Val slices: {len(val_loader.dataset)}")

    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    region_mae  = {name: [0.0, 0] for name in REGION_NAMES.values()}
    n_slices = 0

    for ct, _cbct, mask, region_id in val_loader:
        ct   = ct.to(device)
        mask = mask.to(device)

        mu, _logvar = vae.encode(ct)
        recon = vae.decode(mu).clamp(-1.0, 1.0)

        # mask: (B,1,H,W); apply to recon for metrics
        recon_m = recon * mask + (-1.0) * (1.0 - mask)
        ct_m    = ct    * mask + (-1.0) * (1.0 - mask)

        mae_per = ((_to_hu(recon_m) - _to_hu(ct_m)).abs() * mask).flatten(1).sum(1) \
                  / mask.flatten(1).sum(1).clamp_min(1.0)

        b = ct.size(0)
        total_mae  += float(mae_per.mean()) * b
        total_psnr += _masked_psnr(recon, ct, mask) * b
        total_ssim += _masked_ssim(recon, ct, mask) * b
        n_slices   += b

        for mae_val, rid in zip(mae_per.cpu().tolist(), region_id.tolist()):
            name = REGION_NAMES[int(rid)]
            region_mae[name][0] += mae_val
            region_mae[name][1] += 1

    print("\n=== VAE Reconstruction Baseline (val 23 cases) ===")
    print(f"  mae_hu_vae : {total_mae  / n_slices:.2f} HU")
    print(f"  psnr_vae   : {total_psnr / n_slices:.3f} dB")
    print(f"  ssim_vae   : {total_ssim / n_slices:.4f}")
    print("\nPer-region MAE (HU):")
    for name, (s, c) in sorted(region_mae.items()):
        print(f"  {name}: {s/max(c,1):.2f} HU  (n={c})")
    print()
    print("Interpretation:")
    print("  D1 pilot MAE = 101 HU")
    d1_mae = 101.0
    vae_mae = total_mae / n_slices
    gap = d1_mae - vae_mae
    if vae_mae < 40:
        print(f"  VAE floor = {vae_mae:.1f} HU  → gap to D1 = {gap:.1f} HU  → 扩散模型有大量提升空间，继续训练 D1")
    elif vae_mae < 80:
        print(f"  VAE floor = {vae_mae:.1f} HU  → gap to D1 = {gap:.1f} HU  → 扩散模型有一定提升空间，继续训练值得但 VAE 也构成部分瓶颈")
    else:
        print(f"  VAE floor = {vae_mae:.1f} HU  → gap to D1 = {gap:.1f} HU  → VAE 是主要瓶颈，继续训练扩散模型收益有限，考虑重训 VAE")


if __name__ == "__main__":
    main()
