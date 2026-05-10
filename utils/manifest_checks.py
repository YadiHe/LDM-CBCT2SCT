import json
import logging
from pathlib import Path

import pandas as pd


def validate_manifest_clip(manifest_csv, split, expected_min, expected_max, logger=None, max_files=None):
    """Fail fast if preprocessed data uses a different HU normalization range."""
    log = logger or logging.getLogger(__name__)
    df = pd.read_csv(manifest_csv)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"No rows found in {manifest_csv} for split={split!r}")
    if "preprocess_meta_path" not in df.columns:
        raise ValueError("manifest has no preprocess_meta_path; cannot validate HU clip range.")

    checked = 0
    meta_paths = df["preprocess_meta_path"].dropna()
    if max_files is not None:
        meta_paths = meta_paths.head(max_files)
    for meta_path in meta_paths:
        p = Path(meta_path)
        if not p.exists():
            raise FileNotFoundError(f"preprocess metadata missing: {p}")
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        clip = meta.get("clip")
        if clip is None:
            raise ValueError(f"preprocess metadata has no clip field: {p}")
        got_min, got_max = float(clip[0]), float(clip[1])
        if abs(got_min - expected_min) > 1e-3 or abs(got_max - expected_max) > 1e-3:
            raise ValueError(
                f"HU clip mismatch for {p}: metadata clip={clip}, "
                f"expected=[{expected_min}, {expected_max}]. "
                "Regenerate data/preprocessed and manifest before training/evaluating."
            )
        checked += 1

    if checked:
        log.info(f"validated HU clip range [{expected_min}, {expected_max}] on {checked} metadata files.")
    else:
        log.warning("no preprocess metadata files were checked; HU clip range is unverified.")
