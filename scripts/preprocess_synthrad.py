#!/usr/bin/env python
"""
预处理脚本：synthRAD2023 (Brain) + synthRAD2025 (AB/HN/TH)

流程: HU clip → 前景裁剪 → 等比缩放+中心 Padding → 归一化 → 保存 MHA
输出: data/preprocessed/{pid}/*.mha  +  manifest.csv

用法:
    python scripts/preprocess_synthrad.py \
        --raw-dir  rawdata/ \
        --out-dir  data/preprocessed \
        --manifest data/manifest.csv
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize as sk_resize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CLIP_MIN = -1024.0
CLIP_MAX  =  1500.0
TARGET    = 256
MARGIN    = 10

# Patient-ID prefix → region label
REGION_PREFIXES = [
    ("2ABD", "AB"),
    ("2HN",  "HN"),
    ("2TH",  "TH"),
    ("2BB",  "BB"),
]

# Exact train/val/test counts per region (from dataset analysis)
SPLIT_COUNTS = {
    "BB": (49, 6, 6),
    "AB": (44, 5, 5),
    "HN": (54, 6, 6),
    "TH": (52, 6, 6),
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def detect_region(pid: str) -> str:
    up = pid.upper()
    for prefix, region in REGION_PREFIXES:
        if up.startswith(prefix.upper()):
            return region
    raise ValueError(f"Cannot detect region from patient_id: {pid}")


def find_patients(raw_dir: str):
    """Walk raw_dir and return sorted [(pid, patient_dir)] for dirs with a CT file."""
    result = []
    for root, _, files in os.walk(raw_dir):
        if "ct.nii.gz" in files or "ct.mha" in files:
            result.append((os.path.basename(root), root))
    return sorted(result)


def load_patient(patient_dir: str):
    """
    Load CT, CBCT, mask.
    Returns numpy arrays (D, H, W): ct/cbct float32 HU, mask uint8 binary.
    """
    d = Path(patient_dir)
    ext = ".nii.gz" if (d / "ct.nii.gz").exists() else ".mha"

    def read(name):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(d / (name + ext))))

    ct   = read("ct").astype(np.float32)
    cbct = read("cbct").astype(np.float32)
    mask = read("mask").astype(np.uint8)
    return ct, cbct, mask


def save_vol(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(arr), path, useCompression=True)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def get_xy_bbox(mask: np.ndarray):
    """
    Union bounding box of body mask across all slices, expanded by MARGIN.
    mask shape: (D, H, W), values 0/1.
    Returns (h0, h1, w0, w1) as slice end-points.
    """
    D, H, W = mask.shape
    proj = mask.max(axis=0)          # (H, W)
    rows = np.any(proj, axis=1)
    cols = np.any(proj, axis=0)
    h_idx = np.where(rows)[0]
    w_idx = np.where(cols)[0]
    if len(h_idx) == 0 or len(w_idx) == 0:
        return 0, H, 0, W            # fallback: whole image
    h0 = max(0, h_idx[0]  - MARGIN)
    h1 = min(H, h_idx[-1] + 1 + MARGIN)
    w0 = max(0, w_idx[0]  - MARGIN)
    w1 = min(W, w_idx[-1] + 1 + MARGIN)
    return h0, h1, w0, w1


def resize_vol(arr: np.ndarray, new_H: int, new_W: int, is_mask: bool = False) -> np.ndarray:
    """Resize (D, H, W) in H,W only. Nearest for mask, bilinear for images."""
    D, H, W = arr.shape
    if new_H == H and new_W == W:
        return arr.copy()
    order = 0 if is_mask else 1
    anti  = (not is_mask) and (new_H < H or new_W < W)
    return sk_resize(arr, (D, new_H, new_W),
                     order=order, anti_aliasing=anti,
                     preserve_range=True).astype(arr.dtype)


def center_pad(arr: np.ndarray, target: int, val: float = 0.0) -> np.ndarray:
    """Center-pad (D, H, W) to (D, target, target)."""
    D, H, W = arr.shape
    assert H <= target and W <= target, f"Cannot pad {H}×{W} to {target}×{target}"
    ph, pw = target - H, target - W
    return np.pad(arr,
                  ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                  constant_values=val)


def make_cbct_global(cbct_clipped: np.ndarray) -> np.ndarray:
    """
    Full-volume CBCT (before foreground crop) resized to TARGET×TARGET
    and normalized to [-1, 1]. Saved per-patient for Phase 2 use.
    """
    D, H, W = cbct_clipped.shape
    sc = TARGET / max(H, W)
    r  = resize_vol(cbct_clipped, round(H * sc), round(W * sc))
    r  = center_pad(r, TARGET, CLIP_MIN)
    return ((r - CLIP_MIN) / (CLIP_MAX - CLIP_MIN) * 2.0 - 1.0).astype(np.float32)


def preprocess(ct_raw: np.ndarray, cbct_raw: np.ndarray, mask_raw: np.ndarray):
    """
    Full preprocessing pipeline for one patient volume.

    Returns:
        ct_pp   : (D, 256, 256) float32, normalized [-1, 1]
        cbct_pp : (D, 256, 256) float32, normalized [-1, 1]
        mask_pp : (D, 256, 256) float32, binary {0, 1}
        cbct_global : (D, 256, 256) float32, pre-crop full-body CBCT, normalized
    """
    # Step 2: HU clip
    ct   = np.clip(ct_raw,   CLIP_MIN, CLIP_MAX)
    cbct = np.clip(cbct_raw, CLIP_MIN, CLIP_MAX)

    # Step 6: cbct_global — derived from clipped but UNCROPPED CBCT
    cbct_global = make_cbct_global(cbct)

    # Step 3: foreground crop (XY union bbox of body mask)
    h0, h1, w0, w1 = get_xy_bbox(mask_raw)
    ct   = ct  [:, h0:h1, w0:w1]
    cbct = cbct[:, h0:h1, w0:w1]
    mask = mask_raw[:, h0:h1, w0:w1]

    # Step 4: aspect-ratio resize, then center pad to 256×256
    D, H, W = ct.shape
    sc  = TARGET / max(H, W)
    nH, nW = round(H * sc), round(W * sc)

    ct   = center_pad(resize_vol(ct,                        nH, nW, False), TARGET, CLIP_MIN)
    cbct = center_pad(resize_vol(cbct,                      nH, nW, False), TARGET, CLIP_MIN)
    mf   = center_pad(resize_vol(mask.astype(np.float32),   nH, nW, True),  TARGET, 0.0)
    mask_pp = (mf > 0.5).astype(np.float32)

    # Step 5: normalize to [-1, 1]
    def norm(x):
        return ((x - CLIP_MIN) / (CLIP_MAX - CLIP_MIN) * 2.0 - 1.0).astype(np.float32)

    return norm(ct), norm(cbct), mask_pp, cbct_global


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(by_region: dict) -> list:
    """
    Sort patients per region alphabetically, assign deterministic train/val/test splits.
    Returns list of dicts with keys: patient_id, region, split, _dir
    """
    rows = []
    for region, pats in sorted(by_region.items()):
        pats = sorted(pats, key=lambda x: x[0])
        n_tr, n_vl, n_te = SPLIT_COUNTS[region]
        expected = n_tr + n_vl + n_te
        if len(pats) < expected:
            print(f"  [warn] {region}: found {len(pats)} patients, expected ≥{expected}")
        for i, (pid, pdir) in enumerate(pats[:expected]):
            if i < n_tr:
                split = "train"
            elif i < n_tr + n_vl:
                split = "val"
            else:
                split = "test"
            rows.append({"patient_id": pid, "region": region, "split": split, "_dir": pdir})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SynthRAD preprocessing pipeline")
    p.add_argument("--raw-dir",  required=True,
                   help="Root directory containing patient folders (recursive scan)")
    p.add_argument("--out-dir",  default="data/preprocessed",
                   help="Output directory for preprocessed MHA files")
    p.add_argument("--manifest", default="data/manifest.csv",
                   help="Output manifest CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir  = os.path.abspath(args.out_dir)
    manifest = os.path.abspath(args.manifest)
    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)

    print(f"Scanning {args.raw_dir} for patient directories...")
    all_patients = find_patients(args.raw_dir)
    print(f"Found {len(all_patients)} patients with CT ground truth")

    by_region: dict = {}
    for pid, pdir in all_patients:
        try:
            region = detect_region(pid)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue
        by_region.setdefault(region, []).append((pid, pdir))

    for r in sorted(by_region):
        print(f"  {r}: {len(by_region[r])} patients")

    rows_meta = assign_splits(by_region)
    print(f"\nPreprocessing {len(rows_meta)} patients...")

    manifest_rows = []
    errors = []

    for row in tqdm(rows_meta, desc="preprocessing"):
        pid    = row["patient_id"]
        region = row["region"]
        split  = row["split"]
        pdir   = row["_dir"]
        pid_out = os.path.join(out_dir, pid)

        try:
            ct_raw, cbct_raw, mask_raw = load_patient(pdir)
            ct_pp, cbct_pp, mask_pp, cbct_global = preprocess(ct_raw, cbct_raw, mask_raw)

            ct_path   = os.path.join(pid_out, "ct.mha")
            cbct_path = os.path.join(pid_out, "cbct.mha")
            mask_path = os.path.join(pid_out, "mask.mha")
            glob_path = os.path.join(pid_out, "cbct_global.mha")

            save_vol(ct_pp,      ct_path)
            save_vol(cbct_pp,    cbct_path)
            save_vol(mask_pp,    mask_path)
            save_vol(cbct_global, glob_path)

            manifest_rows.append({
                "patient_id":       pid,
                "region":           region,
                "split":            split,
                "ct_path":          ct_path,
                "cbct_path":        cbct_path,
                "mask_path":        mask_path,
                "cbct_global_path": glob_path,
            })
        except Exception as e:
            print(f"\n  [error] {pid}: {e}")
            errors.append(pid)

    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest, index=False)

    print(f"\nDone. {len(df)} patients saved → {manifest}")
    if errors:
        print(f"  Errors ({len(errors)}): {errors}")
    print("\nSplit summary:")
    print(df.groupby(["region", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
