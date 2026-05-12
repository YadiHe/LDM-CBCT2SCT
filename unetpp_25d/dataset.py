import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


MASK_MIN_PIXELS = 100
REGION_TO_ID = {"BB": 0, "AB": 1, "HN": 2, "TH": 3}


def load_mha(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)


class Slice25DDataset(Dataset):
    """2.5D slice dataset backed by preprocessed 3D MHA volumes.

    Returns:
      cbct: (input_slices, 256, 256), normalized [-1, 1]
      ct:   (1, 256, 256), normalized [-1, 1]
      mask: (1, 256, 256), {0, 1}
      meta: patient/region/z metadata for eval/debug
    """

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        input_slices: int = 7,
        augmentation: bool = False,
        mask_min_pixels: int = MASK_MIN_PIXELS,
    ):
        if input_slices < 1 or input_slices % 2 != 1:
            raise ValueError("input_slices must be a positive odd integer")
        self.input_slices = int(input_slices)
        self.radius = self.input_slices // 2
        self.augmentation = bool(augmentation)
        self.mask_min_pixels = int(mask_min_pixels)

        df = pd.read_csv(manifest_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {manifest_csv}")

        print(f"[Slice25DDataset:{split}] caching {len(df)} volumes into RAM ...")
        self.rows: List[Dict] = []
        self._vols: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        for _, row in df.iterrows():
            ct = load_mha(row["ct_path"])
            cbct = load_mha(row["cbct_path"])
            mask = load_mha(row["mask_path"])
            rid = REGION_TO_ID[str(row["region"])]
            self.rows.append(dict(row))
            self._vols.append((ct, cbct, mask, rid))

        self.index: List[Tuple[int, int]] = []
        for vi, (_, _, mask, _) in enumerate(self._vols):
            for z in range(mask.shape[0]):
                if mask[z].sum() >= self.mask_min_pixels:
                    self.index.append((vi, z))

        print(
            f"[Slice25DDataset:{split}] {len(self.index)} slices "
            f"({len(df)} volumes, input_slices={self.input_slices})"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        vi, z = self.index[idx]
        ct, cbct, mask, rid = self._vols[vi]
        row = self.rows[vi]

        cbct_stack = self.make_cbct_stack(cbct, z)
        ct_t = torch.from_numpy(ct[z]).unsqueeze(0)
        cbct_t = torch.from_numpy(cbct_stack)
        mask_t = torch.from_numpy(mask[z]).unsqueeze(0)

        if self.augmentation:
            cbct_t, ct_t, mask_t = augment(cbct_t, ct_t, mask_t)

        return {
            "cbct": cbct_t,
            "ct": ct_t,
            "mask": mask_t,
            "region_id": torch.tensor(rid, dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "region": str(row["region"]),
            "z": torch.tensor(z, dtype=torch.long),
            "volume_idx": torch.tensor(vi, dtype=torch.long),
        }

    def make_cbct_stack(self, cbct_volume: np.ndarray, z: int) -> np.ndarray:
        depth = cbct_volume.shape[0]
        slices = []
        for dz in range(-self.radius, self.radius + 1):
            zz = int(np.clip(z + dz, 0, depth - 1))
            slices.append(cbct_volume[zz])
        return np.stack(slices, axis=0).astype(np.float32)

    def volume_count(self) -> int:
        return len(self._vols)

    def get_volume(self, volume_idx: int):
        return self._vols[volume_idx], self.rows[volume_idx]

    def foreground_slices(self, volume_idx: int) -> List[int]:
        _, _, mask, _ = self._vols[volume_idx]
        return [z for z in range(mask.shape[0]) if mask[z].sum() >= self.mask_min_pixels]

    def fixed_val_items(self, cases_per_region: int = 4, slices_per_case: int = 3):
        selected = []
        by_region: Dict[str, List[Tuple[str, int, Dict]]] = {}
        for vi, row in enumerate(self.rows):
            by_region.setdefault(str(row["region"]), []).append((str(row["patient_id"]), vi, row))

        for region in sorted(by_region):
            for _, vi, row in sorted(by_region[region])[:cases_per_region]:
                ct, cbct, mask, rid = self._vols[vi]
                z_indices = np.argsort(mask.reshape(mask.shape[0], -1).sum(axis=1))[-slices_per_case:]
                for z in sorted(int(x) for x in z_indices):
                    selected.append({
                        "patient_id": str(row["patient_id"]),
                        "region": region,
                        "z": z,
                        "cbct": torch.from_numpy(self.make_cbct_stack(cbct, z)),
                        "ct": torch.from_numpy(ct[z]).unsqueeze(0),
                        "mask": torch.from_numpy(mask[z]).unsqueeze(0),
                    })
        return selected


def augment(cbct: torch.Tensor, ct: torch.Tensor, mask: torch.Tensor):
    if random.random() < 0.5:
        cbct, ct, mask = TF.hflip(cbct), TF.hflip(ct), TF.hflip(mask)
    if random.random() < 0.3:
        cbct, ct, mask = TF.vflip(cbct), TF.vflip(ct), TF.vflip(mask)
    if random.random() < 0.5:
        angle = random.uniform(-5.0, 5.0)
        cbct = TF.rotate(cbct, angle, fill=-1.0)
        ct = TF.rotate(ct, angle, fill=-1.0)
        mask = TF.rotate(mask, angle, fill=0.0, interpolation=TF.InterpolationMode.NEAREST)
    return cbct, ct, mask


def collate_keep_meta(batch: List[Dict]):
    tensor_keys = ["cbct", "ct", "mask", "region_id", "z", "volume_idx"]
    out = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    out["patient_id"] = [item["patient_id"] for item in batch]
    out["region"] = [item["region"] for item in batch]
    return out


def seed_worker(seed: Optional[int]):
    if seed is None:
        return None

    def _worker_init(worker_id):
        worker_seed = (seed + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init


def get_dataloaders(
    manifest_csv: str,
    batch_size: int,
    input_slices: int = 7,
    num_workers: int = 4,
    augmentation: bool = True,
    seed: Optional[int] = None,
    mask_min_pixels: int = MASK_MIN_PIXELS,
):
    train_ds = Slice25DDataset(
        manifest_csv, "train", input_slices=input_slices,
        augmentation=augmentation, mask_min_pixels=mask_min_pixels,
    )
    val_ds = Slice25DDataset(
        manifest_csv, "val", input_slices=input_slices,
        augmentation=False, mask_min_pixels=mask_min_pixels,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker(seed),
        collate_fn=collate_keep_meta,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=generator, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common,
    )
    return train_loader, val_loader
