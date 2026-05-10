#!/usr/bin/env python
"""
预处理脚本：synthRAD2023 (Brain) + synthRAD2025 (AB/HN/TH)

流程: 一次性加载 → HU clip → 前景裁剪 → mask 外置空气 → 等比缩放+中心 Padding → 归一化 → 保存 MHA
输出: data/preprocessed/{pid}/*.mha  +  manifest.csv

用法:
    python scripts/preprocess_synthrad_dataset.py \
        --raw-dir  rawdata/ \
        --out-dir  data/preprocessed \
        --manifest data/manifest.csv
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ITK 5.4.x still touches np.bool during import under numpy 1.24.
if "bool" not in np.__dict__:
    np.bool = bool

try:
    from monai.transforms import (
        Compose,
        EnsureTyped,
        ScaleIntensityRanged,
        MapTransform,
        Resized,
        SpatialPadd,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        'MONAI is required for preprocessing. Install it with: pip install "monai[itk,nibabel]"'
    ) from e

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.hu import CLIP_MIN, CLIP_MAX  # single source of truth

TARGET = 256
MARGIN = 10

# Patient-ID prefix → region label
REGION_PREFIXES = [
    ("2ABD", "AB"),
    ("2HN",  "HN"),
    ("2TH",  "TH"),
    ("2BB",  "BB"),
]

# Per-region (n_train_first, n_val) counts. Patients are sorted alphabetically;
# the first n_train_first go to train, the next n_val to val, and any remainder
# also goes to train. Region totals: BB=60, AB=53, HN=65, TH=63 → 218 train + 23 val.
# The official challenge val zip (no CT GT) is not handled by this script and is
# reserved for post-training inference / submission.
SPLIT_COUNTS = {
    "BB": (49, 6),
    "AB": (44, 5),
    "HN": (54, 6),
    "TH": (52, 6),
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


def get_patient_paths(patient_dir: str) -> dict:
    """Return CT/CBCT/mask paths for one patient directory."""
    d = Path(patient_dir)
    ext = ".nii.gz" if (d / "ct.nii.gz").exists() else ".mha"
    return {
        "ct": str(d / ("ct" + ext)),
        "cbct": str(d / ("cbct" + ext)),
        "mask": str(d / ("mask" + ext)),
    }


def load_patient_volumes(patient_paths: dict):
    """
    Read CT/CBCT/mask once, validate same voxel grid, and return:
        arrays   : dict[str, np.ndarray] of shape (1, Z, H, W) in channel-first
        geometry : dict with x/y/z spacing/origin/direction/size from CT

    Replaces previous read_geometry + validate_same_geometry + LoadSITKArrayd
    chain (eliminates 6+ redundant SimpleITK reads per patient).
    """
    images = {k: sitk.ReadImage(p) for k, p in patient_paths.items()}
    ref = images["ct"]
    ref_geom = (ref.GetSize(), ref.GetSpacing(), ref.GetOrigin(), ref.GetDirection())
    for k, img in images.items():
        if k == "ct":
            continue
        if (img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection()) != ref_geom:
            raise ValueError(
                f"{k} geometry does not match CT: "
                f"ct(size={ref.GetSize()}, spacing={ref.GetSpacing()}, origin={ref.GetOrigin()}) vs "
                f"{k}(size={img.GetSize()}, spacing={img.GetSpacing()}, origin={img.GetOrigin()})"
            )
    arrays = {k: sitk.GetArrayFromImage(img)[None] for k, img in images.items()}
    geometry = {
        "spacing": list(ref.GetSpacing()),
        "origin": list(ref.GetOrigin()),
        "direction": list(ref.GetDirection()),
        "size": list(ref.GetSize()),
    }
    return arrays, geometry


def save_vol(arr: np.ndarray, path: str, geometry: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(geometry["spacing"]))
    img.SetOrigin(tuple(geometry["origin"]))
    img.SetDirection(tuple(geometry["direction"]))
    sitk.WriteImage(img, path, True)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_xy_bbox(mask: np.ndarray):
    """
    Union bounding box of body mask across all slices, expanded by MARGIN.
    mask shape: (D, H, W), values 0/1.
    Returns (h0, h1, w0, w1) as slice end-points.
    """
    D, H, W = mask.shape
    proj = mask.max(axis=0)
    rows = np.any(proj, axis=1)
    cols = np.any(proj, axis=0)
    h_idx = np.where(rows)[0]
    w_idx = np.where(cols)[0]
    if len(h_idx) == 0 or len(w_idx) == 0:
        return 0, H, 0, W
    h0 = max(0, h_idx[0]  - MARGIN)
    h1 = min(H, h_idx[-1] + 1 + MARGIN)
    w0 = max(0, w_idx[0]  - MARGIN)
    w1 = min(W, w_idx[-1] + 1 + MARGIN)
    return h0, h1, w0, w1


class MaskForegroundCropd(MapTransform):
    """Crop all keys to the XY union bbox of mask foreground."""

    def __init__(self, keys, source_key="mask", margin=10):
        super().__init__(keys)
        self.source_key = source_key
        self.margin = margin

    def __call__(self, data):
        d = dict(data)
        mask = _as_numpy(d[self.source_key][0])
        h0, h1, w0, w1 = get_xy_bbox(mask)
        for key in self.keys:
            d[key] = d[key][..., h0:h1, w0:w1]
        d["preprocess_meta"]["crop_bbox_xy"] = {
            "h0": int(h0), "h1": int(h1), "w0": int(w0), "w1": int(w1),
        }
        return d


class SetAirBackgroundd(MapTransform):
    """Set voxels outside `mask_key` foreground to `air_val` for each key in `keys`."""

    def __init__(self, keys, mask_key="mask", air_val=-1024.0):
        super().__init__(keys)
        self.mask_key = mask_key
        self.air_val = float(air_val)

    def __call__(self, data):
        d = dict(data)
        # mask is (1, Z, H, W); broadcasts directly against image (1, Z, H, W).
        body = d[self.mask_key] > 0.5
        for key in self.keys:
            img = d[key]
            d[key] = torch.where(body, img, img.new_full((), self.air_val))
        return d


class ResizeWithAspectRatioAndPadd(MapTransform):
    """Resize long side to target and center-pad short side to target."""

    def __init__(self, keys, target_size=256, pad_value=-1024.0, mask_key="mask"):
        super().__init__(keys)
        self.target = target_size
        self.pad_value = pad_value
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        H, W = int(d[self.keys[0]].shape[-2]), int(d[self.keys[0]].shape[-1])
        scale = self.target / max(H, W)
        new_H, new_W = round(H * scale), round(W * scale)
        pad_h, pad_w = self.target - new_H, self.target - new_W

        img_keys  = [k for k in self.keys if k != self.mask_key]
        mask_keys = [k for k in self.keys if k == self.mask_key]

        if img_keys:
            d = Resized(
                keys=img_keys,
                spatial_size=(-1, new_H, new_W),
                mode="bilinear",
                anti_aliasing=(scale < 1.0),
            )(d)
            d = SpatialPadd(
                keys=img_keys,
                spatial_size=(-1, self.target, self.target),
                mode="constant",
                constant_values=self.pad_value,
            )(d)
        if mask_keys:
            d = Resized(
                keys=mask_keys,
                spatial_size=(-1, new_H, new_W),
                mode="nearest",
            )(d)
            d = SpatialPadd(
                keys=mask_keys,
                spatial_size=(-1, self.target, self.target),
                mode="constant",
                constant_values=0,
            )(d)

        d["preprocess_meta"]["resize_pad"] = {
            "target": int(self.target),
            "crop_shape_hw": [int(H), int(W)],
            "resized_shape_hw": [int(new_H), int(new_W)],
            "scale": float(scale),
            "pad_top": int(pad_h // 2),
            "pad_bottom": int(pad_h - pad_h // 2),
            "pad_left": int(pad_w // 2),
            "pad_right": int(pad_w - pad_w // 2),
        }
        return d


class BinarizeMaskd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0.5).to(torch.float32)
        return d


def build_preprocess_transform():
    """
    Local pipeline: clip → crop to body bbox → set background to air →
    resize+pad → re-apply air background → normalize.

    Two SetAirBackground passes:
      - pre-resize:  zeros out non-body voxels (table/scatter/artifacts) in the
                     original grid before bilinear resize blurs them inward.
      - post-resize: cleans the 1-2 voxel rim where bilinear (image) and nearest
                     (mask) resampling disagree at sub-pixel level, leaving a
                     few hundred ppm of "bleed" voxels with non-air CT.

    Inputs come in as channel-first (1, Z, H, W) numpy arrays — already loaded
    by `load_patient_volumes`, so no IO transform is needed at the head.
    """
    return Compose([
        EnsureTyped(keys=["ct", "cbct"], dtype=torch.float32),
        EnsureTyped(keys=["mask"], dtype=torch.uint8),
        ScaleIntensityRanged(
            keys=["ct", "cbct"],
            a_min=CLIP_MIN, a_max=CLIP_MAX,
            b_min=CLIP_MIN, b_max=CLIP_MAX,
            clip=True,
        ),
        MaskForegroundCropd(keys=["ct", "cbct", "mask"], source_key="mask", margin=MARGIN),
        SetAirBackgroundd(keys=["ct", "cbct"], mask_key="mask", air_val=CLIP_MIN),
        ResizeWithAspectRatioAndPadd(
            keys=["ct", "cbct", "mask"],
            target_size=TARGET,
            pad_value=CLIP_MIN,
            mask_key="mask",
        ),
        SetAirBackgroundd(keys=["ct", "cbct"], mask_key="mask", air_val=CLIP_MIN),
        ScaleIntensityRanged(
            keys=["ct", "cbct"],
            a_min=CLIP_MIN, a_max=CLIP_MAX,
            b_min=-1.0,     b_max=1.0,
            clip=True,
        ),
        BinarizeMaskd(keys=["mask"]),
    ])


def build_global_transform():
    """
    Global pipeline: same as local minus the body bbox crop, used to feed full-FOV
    CBCT into the global ControlNet branch. Mask is co-resized with CBCT so the
    post-resize SetAirBackground can clean bilinear edge bleed.
    """
    return Compose([
        EnsureTyped(keys=["cbct"], dtype=torch.float32),
        EnsureTyped(keys=["mask"], dtype=torch.uint8),
        ScaleIntensityRanged(
            keys=["cbct"],
            a_min=CLIP_MIN, a_max=CLIP_MAX,
            b_min=CLIP_MIN, b_max=CLIP_MAX,
            clip=True,
        ),
        SetAirBackgroundd(keys=["cbct"], mask_key="mask", air_val=CLIP_MIN),
        ResizeWithAspectRatioAndPadd(
            keys=["cbct", "mask"],
            target_size=TARGET,
            pad_value=CLIP_MIN,
            mask_key="mask",
        ),
        SetAirBackgroundd(keys=["cbct"], mask_key="mask", air_val=CLIP_MIN),
        ScaleIntensityRanged(
            keys=["cbct"],
            a_min=CLIP_MIN, a_max=CLIP_MAX,
            b_min=-1.0,     b_max=1.0,
            clip=True,
        ),
    ])


def output_geometry(original: dict, meta: dict) -> dict:
    """Compute geometry for cropped/resized/padded output in original orientation."""
    bbox = meta["crop_bbox_xy"]
    rp = meta["resize_pad"]
    scale = rp["scale"]
    in_spacing = original["spacing"]
    out_spacing = [float(in_spacing[0] / scale), float(in_spacing[1] / scale), float(in_spacing[2])]

    direction = np.asarray(original["direction"], dtype=np.float64).reshape(3, 3)
    offset_index = np.asarray([
        bbox["w0"] * in_spacing[0] - rp["pad_left"] * out_spacing[0],
        bbox["h0"] * in_spacing[1] - rp["pad_top"] * out_spacing[1],
        0.0,
    ])
    origin = np.asarray(original["origin"], dtype=np.float64) + direction.dot(offset_index)

    return {
        "spacing": out_spacing,
        "origin": origin.tolist(),
        "direction": original["direction"],
    }


def restore_preprocessed_to_original(
    arr: np.ndarray,
    preprocess_meta: dict,
    fill_value: float = -1.0,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Restore a preprocessed (Z, 256, 256) volume to the original (Z, H, W) grid.

    Intended for inference/QC symmetry: remove center padding, resize back to the
    mask-crop shape, then paste into the original XY canvas.
    """
    rp = preprocess_meta["resize_pad"]
    bbox = preprocess_meta["crop_bbox_xy"]
    orig_size = preprocess_meta["original_geometry"]["size"]
    orig_w, orig_h, orig_z = int(orig_size[0]), int(orig_size[1]), int(orig_size[2])
    crop_h, crop_w = rp["crop_shape_hw"]

    if arr.shape[0] != orig_z:
        raise ValueError(f"Z mismatch: preprocessed has {arr.shape[0]}, original metadata has {orig_z}")

    h0 = rp["pad_top"]
    h1 = arr.shape[1] - rp["pad_bottom"]
    w0 = rp["pad_left"]
    w1 = arr.shape[2] - rp["pad_right"]
    cropped = arr[:, h0:h1, w0:w1]

    mode = "nearest" if is_mask else "bilinear"
    t = torch.from_numpy(cropped.astype(np.float32)).unsqueeze(1)
    kwargs = {} if is_mask else {"align_corners": False}
    restored_crop = F.interpolate(t, size=(crop_h, crop_w), mode=mode, **kwargs).squeeze(1).numpy()

    restored = np.full((orig_z, orig_h, orig_w), fill_value, dtype=np.float32)
    restored[:, bbox["h0"]:bbox["h1"], bbox["w0"]:bbox["w1"]] = restored_crop
    if is_mask:
        restored = (restored > 0.5).astype(np.float32)
    return restored


def preprocess(arrays: dict, original_geometry: dict):
    """
    Full preprocessing pipeline for one patient's already-loaded volumes.

    Args:
        arrays: dict with keys "ct", "cbct", "mask", each shape (1, Z, H, W).
        original_geometry: x/y/z spacing/origin/direction/size from the source CT.

    Returns:
        ct_pp       : (Z, 256, 256) float32, normalized [-1, 1]
        cbct_pp     : (Z, 256, 256) float32, normalized [-1, 1]
        mask_pp     : (Z, 256, 256) float32, binary {0, 1}
        cbct_global : (Z, 256, 256) float32, full-FOV CBCT, normalized [-1, 1]
        meta        : preprocessing metadata (crop bbox, resize/pad, output geometries)
    """
    base_meta = {
        "clip": [CLIP_MIN, CLIP_MAX],
        "margin": MARGIN,
        "original_geometry": original_geometry,
    }

    local_data = {
        "ct": arrays["ct"],
        "cbct": arrays["cbct"],
        "mask": arrays["mask"],
        "preprocess_meta": dict(base_meta),
    }
    local = build_preprocess_transform()(local_data)

    orig_w, orig_h = int(original_geometry["size"][0]), int(original_geometry["size"][1])
    global_data = {
        "cbct": arrays["cbct"],
        "mask": arrays["mask"],
        "preprocess_meta": {
            **base_meta,
            "margin": 0,
            "crop_bbox_xy": {"h0": 0, "h1": orig_h, "w0": 0, "w1": orig_w},
        },
    }
    global_pp = build_global_transform()(global_data)

    meta = local["preprocess_meta"]
    meta["output_geometry"] = output_geometry(original_geometry, meta)
    meta["global_resize_pad"] = global_pp["preprocess_meta"]["resize_pad"]
    meta["global_output_geometry"] = output_geometry(
        original_geometry,
        {
            "crop_bbox_xy": global_data["preprocess_meta"]["crop_bbox_xy"],
            "resize_pad":   global_pp["preprocess_meta"]["resize_pad"],
        },
    )

    def vol(d, key):
        return _as_numpy(d[key][0]).astype(np.float32)

    return (
        vol(local, "ct"),
        vol(local, "cbct"),
        vol(local, "mask"),
        vol(global_pp, "cbct"),
        meta,
    )


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(by_region: dict) -> list:
    """
    Sort patients per region alphabetically and assign deterministic train/val splits.

    Indices [0, n_tr) → train, [n_tr, n_tr+n_vl) → val, [n_tr+n_vl, end) → train.
    The trailing-train segment absorbs whatever remains after train+val so that no
    GT-bearing patient is dropped.
    """
    rows = []
    for region, pats in sorted(by_region.items()):
        pats = sorted(pats, key=lambda x: x[0])
        n_tr, n_vl = SPLIT_COUNTS[region]
        if len(pats) < n_tr + n_vl:
            print(f"  [warn] {region}: found {len(pats)} patients, expected ≥{n_tr + n_vl}")
        for i, (pid, pdir) in enumerate(pats):
            split = "val" if (n_tr <= i < n_tr + n_vl) else "train"
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


def _verify_air_background(ct_path: str, mask_path: str, pid: str, tol: float = 0.05) -> bool:
    """Read back saved CT/mask and check mask==0 voxels are normalized air (≈ -1).

    Returns True iff `max(bg) <= -1 + tol`. Prints a [check]/[verify-warn] line.
    """
    ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
    mk = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
    bg = ct[mk < 0.5]
    if bg.size == 0:
        print(f"  [check] {pid}: mask is full-coverage (no background voxels)")
        return True
    bg_min = float(bg.min())
    bg_max = float(bg.max())
    if bg_max <= -1.0 + tol:
        print(f"  [check] {pid}: background CT min={bg_min:.4f}, max={bg_max:.4f} (≈ -1.0 ✓)")
        return True
    print(f"  [verify-warn] {pid}: background CT max={bg_max:.4f} > -1+{tol} (file kept; review)")
    return False


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
    verify_warns = []  # patients whose post-save background check did not pass cleanly

    for row in tqdm(rows_meta, desc="preprocessing"):
        pid    = row["patient_id"]
        region = row["region"]
        split  = row["split"]
        pdir   = row["_dir"]
        pid_out = os.path.join(out_dir, pid)

        try:
            patient_paths = get_patient_paths(pdir)
            arrays, original_geometry = load_patient_volumes(patient_paths)
            ct_pp, cbct_pp, mask_pp, cbct_global, pp_meta = preprocess(arrays, original_geometry)

            os.makedirs(pid_out, exist_ok=True)
            ct_path   = os.path.join(pid_out, "ct_preprocessed.mha")
            cbct_path = os.path.join(pid_out, "cbct_preprocessed.mha")
            mask_path = os.path.join(pid_out, "mask_preprocessed.mha")
            glob_path = os.path.join(pid_out, "cbct_global.mha")
            meta_path = os.path.join(pid_out, "preprocess_metadata.json")

            save_vol(ct_pp,       ct_path,   pp_meta["output_geometry"])
            save_vol(cbct_pp,     cbct_path, pp_meta["output_geometry"])
            save_vol(mask_pp,     mask_path, pp_meta["output_geometry"])
            save_vol(cbct_global, glob_path, pp_meta["global_output_geometry"])
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(pp_meta, f, indent=2)

            manifest_rows.append({
                "patient_id":           pid,
                "region":               region,
                "split":                split,
                "ct_path":              ct_path,
                "cbct_path":            cbct_path,
                "mask_path":            mask_path,
                "cbct_global_path":     glob_path,
                "preprocess_meta_path": meta_path,
            })
        except Exception as e:
            print(f"\n  [error] {pid}: {e}")
            errors.append(pid)
            continue

        # Post-save QC: warn but never invalidate the manifest entry. Verify only the
        # first successful patient per region — enough to catch regional regressions.
        if region not in {r["region"] for r in manifest_rows[:-1]}:
            if not _verify_air_background(ct_path, mask_path, pid):
                verify_warns.append(pid)

    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest, index=False)

    print(f"\nDone. {len(df)} patients saved → {manifest}")
    if errors:
        print(f"  Errors ({len(errors)}): {errors}")
    if verify_warns:
        print(f"  Verify warns ({len(verify_warns)}): {verify_warns}")
    print("\nSplit summary:")
    print(df.groupby(["region", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
