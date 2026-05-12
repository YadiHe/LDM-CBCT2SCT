import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from unetpp_25d.dataset import Slice25DDataset
from unetpp_25d.model import UNetPlusPlus25D
from utils.hu import to_hu
from utils.image_metrics import ImageMetrics


@torch.no_grad()
def predict_patient(
    model,
    dataset: Slice25DDataset,
    volume_idx: int,
    device,
    batch_size: int = 32,
    amp_enabled: bool = True,
    amp_dtype: str = "fp16",
):
    (ct, cbct, mask, _rid), row = dataset.get_volume(volume_idx)
    pred = np.full_like(ct, fill_value=-1.0, dtype=np.float32)
    z_list = dataset.foreground_slices(volume_idx)
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16

    for start in range(0, len(z_list), batch_size):
        zs = z_list[start:start + batch_size]
        x = np.stack([dataset.make_cbct_stack(cbct, z) for z in zs], axis=0)
        x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=dtype):
            y = model(x_t)
        y_np = y.detach().float().cpu().numpy()[:, 0]
        for z, y_slice in zip(zs, y_np):
            pred[z] = y_slice

    pred = np.where(mask > 0.5, pred, -1.0).astype(np.float32)
    return ct, pred, mask, row


@torch.no_grad()
def evaluate_model(
    model,
    dataset: Slice25DDataset,
    device,
    batch_size: int = 32,
    amp_enabled: bool = True,
    amp_dtype: str = "fp16",
    max_patients: Optional[int] = None,
    prefix: str = "val",
):
    model.eval()
    metrics_obj = ImageMetrics()
    totals = {"mae_hu": 0.0, "psnr": 0.0, "ms_ssim": 0.0}
    region_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "mae_hu": 0.0, "psnr": 0.0, "ms_ssim": 0.0, "n": 0,
    })
    n = 0

    volume_indices = range(dataset.volume_count())
    if max_patients is not None:
        volume_indices = list(volume_indices)[:max_patients]

    for vi in volume_indices:
        ct_norm, pred_norm, mask, row = predict_patient(
            model, dataset, vi, device, batch_size=batch_size,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        ct_hu = to_hu(ct_norm)
        pred_hu = to_hu(pred_norm)
        scores = metrics_obj.score_patient(ct_hu, pred_hu, mask)
        mae = float(scores["mae"])
        psnr = float(scores["psnr"])
        ms_ssim = float(scores["ms_ssim"])
        totals["mae_hu"] += mae
        totals["psnr"] += psnr
        totals["ms_ssim"] += ms_ssim
        region = str(row["region"])
        region_totals[region]["mae_hu"] += mae
        region_totals[region]["psnr"] += psnr
        region_totals[region]["ms_ssim"] += ms_ssim
        region_totals[region]["n"] += 1
        n += 1

    out = {
        f"{prefix}/mae_hu": totals["mae_hu"] / max(n, 1),
        f"{prefix}/psnr": totals["psnr"] / max(n, 1),
        f"{prefix}/ms_ssim": totals["ms_ssim"] / max(n, 1),
        f"{prefix}/n_patients": n,
    }
    for region, vals in sorted(region_totals.items()):
        rn = max(int(vals["n"]), 1)
        out[f"{prefix}/mae_hu_{region}"] = vals["mae_hu"] / rn
        out[f"{prefix}/psnr_{region}"] = vals["psnr"] / rn
        out[f"{prefix}/ms_ssim_{region}"] = vals["ms_ssim"] / rn
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate 2.5D U-Net++ on patient-level metrics")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-slices", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--encoder-name", default="resnet34")
    p.add_argument("--encoder-weights", default="none")
    p.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--max-patients", type=int, default=None)
    args = p.parse_args()

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(device_name)
    ds = Slice25DDataset(args.manifest, "val", input_slices=args.input_slices, augmentation=False)
    model = UNetPlusPlus25D(
        in_channels=args.input_slices,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    metrics = evaluate_model(
        model, ds, device, batch_size=args.batch_size,
        amp_enabled=device.type == "cuda" and not args.no_amp,
        amp_dtype=args.amp_dtype, max_patients=args.max_patients,
    )
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
