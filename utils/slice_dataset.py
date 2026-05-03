"""
SliceDataset: 2D slice dataset backed by MHA volumes cached in RAM.

每次 __getitem__ 返回一个 256×256 切片 (ct, cbct, mask, region_id)，
所有 volume 在构造时一次性读入内存（~12 GB << 1007 GB）。
"""
import os
import random

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

MASK_MIN_PIXELS = 100          # 过滤几乎全是空气的切片
REGION_TO_ID = {"BB": 0, "AB": 1, "HN": 2, "TH": 3}


class SliceDataset(Dataset):
    """
    从 manifest CSV 加载预处理后的 MHA volume，展开为 2D slice。

    Manifest 必须包含列：patient_id, region, split, ct_path, cbct_path, mask_path

    __getitem__ 返回:
        ct        : (1, 256, 256) float32, 归一化 [-1, 1]
        cbct      : (1, 256, 256) float32, 归一化 [-1, 1]
        mask      : (1, 256, 256) float32, {0.0, 1.0}
        region_id : () long, 0=BB / 1=AB / 2=HN / 3=TH
    """

    def __init__(self, manifest_csv: str, split: str, augmentation: bool = False):
        df = pd.read_csv(manifest_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {manifest_csv}")

        print(f"[SliceDataset:{split}] caching {len(df)} volumes into RAM ...")
        self.rows: list = []
        self._vols: list = []
        for _, row in df.iterrows():
            ct   = _load_mha(row["ct_path"])     # (D, 256, 256) float32
            cbct = _load_mha(row["cbct_path"])
            mask = _load_mha(row["mask_path"])   # float32 {0, 1}
            rid  = REGION_TO_ID[row["region"]]
            self._vols.append((ct, cbct, mask, rid))
            self.rows.append(dict(row))

        # Build (volume_idx, slice_z) index, skip near-air slices
        self.index: list = []
        for vi, (_, _, mask, _) in enumerate(self._vols):
            for z in range(mask.shape[0]):
                if mask[z].sum() >= MASK_MIN_PIXELS:
                    self.index.append((vi, z))

        print(f"[SliceDataset:{split}] {len(self.index)} slices "
              f"({len(df)} volumes, {len(self._vols[0][0])} slices/vol approx)")

        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        vi, z = self.index[idx]
        ct, cbct, mask, rid = self._vols[vi]

        ct_t   = torch.from_numpy(ct  [z]).unsqueeze(0)   # (1, 256, 256)
        cbct_t = torch.from_numpy(cbct[z]).unsqueeze(0)
        mask_t = torch.from_numpy(mask[z]).unsqueeze(0)
        rid_t  = torch.tensor(rid, dtype=torch.long)

        if self.augmentation:
            ct_t, cbct_t, mask_t = _augment(ct_t, cbct_t, mask_t)

        return ct_t, cbct_t, mask_t, rid_t

    def fixed_val_items(self, cases_per_region: int = 4, slices_per_case: int = 3):
        """Return deterministic fixed-val slices: first patients per region, top mask-area slices."""
        selected = []
        by_region = {}
        for vi, row in enumerate(self.rows):
            by_region.setdefault(row["region"], []).append((str(row["patient_id"]), vi, row))

        for region in sorted(by_region):
            for _, vi, row in sorted(by_region[region])[:cases_per_region]:
                ct, cbct, mask, rid = self._vols[vi]
                z_indices = np.argsort(mask.reshape(mask.shape[0], -1).sum(axis=1))[-slices_per_case:]
                z_indices = sorted(int(z) for z in z_indices)
                for z in z_indices:
                    selected.append({
                        "patient_id": str(row["patient_id"]),
                        "region": region,
                        "z": z,
                        "ct": torch.from_numpy(ct[z]).unsqueeze(0),
                        "cbct": torch.from_numpy(cbct[z]).unsqueeze(0),
                        "mask": torch.from_numpy(mask[z]).unsqueeze(0),
                        "region_id": torch.tensor(rid, dtype=torch.long),
                    })
        return selected


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_mha(path: str) -> np.ndarray:
    """Load MHA file → (D, H, W) float32 numpy array."""
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _augment(ct: torch.Tensor, cbct: torch.Tensor, mask: torch.Tensor):
    """Synchronized random flip + rotation. All tensors (1, H, W)."""
    if random.random() < 0.5:
        ct, cbct, mask = TF.hflip(ct), TF.hflip(cbct), TF.hflip(mask)
    if random.random() < 0.3:
        ct, cbct, mask = TF.vflip(ct), TF.vflip(cbct), TF.vflip(mask)
    if random.random() < 0.5:
        angle = random.uniform(-5.0, 5.0)
        ct   = TF.rotate(ct,   angle, fill=-1.0)
        cbct = TF.rotate(cbct, angle, fill=-1.0)
        mask = TF.rotate(mask, angle, fill=0.0,
                         interpolation=TF.InterpolationMode.NEAREST)
    return ct, cbct, mask


def get_dataloaders(
    manifest_csv: str,
    batch_size: int,
    num_workers: int = 4,
    augmentation: bool = True,
    seed: int = None,
):
    """
    Build train and val DataLoaders from a manifest CSV.

    Returns:
        train_loader, val_loader
    """
    train_ds = SliceDataset(manifest_csv, "train", augmentation=augmentation)
    val_ds   = SliceDataset(manifest_csv, "val",   augmentation=False)

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def worker_init_fn(worker_id):
            worker_seed = (seed + worker_id) % 2**32
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader
