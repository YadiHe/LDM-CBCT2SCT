"""Patient-level HU-space image metrics for SynthRAD CBCT→sCT.

Math and defaults are kept identical to the SynthRAD2025 reference
implementation so internal numbers can be compared against the challenge
leaderboard. In particular:

- Dynamic range is fixed at [-1024, 3000] (SynthRAD population range), NOT
  the project's v2 CLIP range [-1024, 2000]. PSNR uses this as data_range,
  MS-SSIM uses it for K1*R / K2*R constants and for the [0, R] shift.
- MS-SSIM is the masked variant: SSIM map is computed full-image at each
  scale, then mean-reduced over the mask only. Volumes smaller than 97 in
  any axis are edge-padded to 97 before scoring.
- The official `luminance_weights = [0, 0, 0, 0, 0, 0.1333]` indexing oddity
  is preserved verbatim — mirrors the reference repository.

Inputs to `score_patient` are HU-space 3D numpy arrays. Use
`utils.hu.to_hu` to denormalize model outputs before calling this module.
"""
from typing import Optional

import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio
from skimage.util.arraycrop import crop
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter, gaussian_filter as gaussian


class ImageMetrics:
    """SynthRAD-aligned MAE / PSNR / masked MS-SSIM in HU space."""

    def __init__(self, debug: bool = False):
        # SynthRAD official population-wide dynamic range
        self.dynamic_range = [-1024.0, 3000.0]
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_patient(self, gt_img: np.ndarray, synthetic_ct: np.ndarray,
                      mask: Optional[np.ndarray]) -> dict:
        """Return {'mae', 'psnr', 'ms_ssim'} for one HU-space 3D volume."""
        assert gt_img.shape == synthetic_ct.shape
        if mask is not None:
            assert mask.shape == synthetic_ct.shape

        ground_truth = gt_img if mask is None else np.where(mask == 0, -1024, gt_img)
        prediction   = synthetic_ct if mask is None else np.where(mask == 0, -1024, synthetic_ct)

        mae_value = self.mae(ground_truth, prediction, mask)
        psnr_value = self.psnr(ground_truth, prediction, mask, use_population_range=True)
        _ms_full, ms_ssim_mask = self.ms_ssim(ground_truth, prediction, mask)

        return {
            "mae": mae_value,
            "psnr": psnr_value,
            "ms_ssim": ms_ssim_mask,
        }

    def score_patient_from_paths(self, gt_path: str, pred_path: str,
                                 mask_path: Optional[str]) -> dict:
        """Convenience: load three MHA volumes and call score_patient."""
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.float32)
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).astype(np.float32)
        mask = (sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
                if mask_path is not None else None)
        return self.score_patient(gt, pred, mask)

    # ------------------------------------------------------------------
    # MAE
    # ------------------------------------------------------------------

    def mae(self, gt: np.ndarray, pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """Mean absolute error in HU space, restricted to mask voxels."""
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            mask = np.where(mask > 0, 1.0, 0.0)
        mae_value = np.sum(np.abs(gt * mask - pred * mask)) / mask.sum()
        return float(mae_value)

    # ------------------------------------------------------------------
    # PSNR
    # ------------------------------------------------------------------

    def psnr(self, gt: np.ndarray, pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: bool = False) -> float:
        """Peak SNR, optionally with population-wide dynamic range."""
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            mask = np.where(mask > 0, 1.0, 0.0)

        if use_population_range:
            gt = np.clip(gt, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            pred = np.clip(pred, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        else:
            dynamic_range = gt.max() - gt.min()
            pred = np.clip(pred, a_min=gt.min(), a_max=gt.max())

        gt = gt[mask == 1]
        pred = pred[mask == 1]
        return float(peak_signal_noise_ratio(gt, pred, data_range=dynamic_range))

    # ------------------------------------------------------------------
    # SSIM components and MS-SSIM
    # ------------------------------------------------------------------

    def structural_similarity_at_scale(
        self, im1, im2, *,
        luminance_weight=1, contrast_weight=1, structure_weight=1,
        win_size=None, gradient=False, data_range=None,
        channel_axis=None, gaussian_weights=False, full=False, **kwargs,
    ):
        """Per-scale SSIM with separable luminance / contrast / structure weights.

        Faithful to the SynthRAD reference; only the local `gaussian` symbol
        is bound to scipy's gaussian_filter so the gaussian_weights branch
        works (the reference left it unbound).
        """
        K1 = kwargs.pop("K1", 0.01)
        K2 = kwargs.pop("K2", 0.03)
        sigma = kwargs.pop("sigma", 1.5)
        if K1 < 0:
            raise ValueError("K1 must be positive")
        if K2 < 0:
            raise ValueError("K2 must be positive")
        if sigma < 0:
            raise ValueError("sigma must be positive")
        use_sample_covariance = kwargs.pop("use_sample_covariance", True)

        if gaussian_weights:
            truncate = 3.5
        if win_size is None:
            if gaussian_weights:
                r = int(truncate * sigma + 0.5)
                win_size = 2 * r + 1
            else:
                win_size = 7
        if gaussian_weights:
            filter_func = gaussian
            filter_args = {"sigma": sigma, "truncate": truncate, "mode": "reflect"}
        else:
            filter_func = uniform_filter
            filter_args = {"size": win_size}

        ndim = im1.ndim
        NP = win_size ** ndim
        cov_norm = NP / (NP - 1) if use_sample_covariance else 1.0

        ux = filter_func(im1, **filter_args)
        uy = filter_func(im2, **filter_args)
        uxx = filter_func(im1 * im1, **filter_args)
        uyy = filter_func(im2 * im2, **filter_args)
        uxy = filter_func(im1 * im2, **filter_args)
        vx = cov_norm * (uxx - ux * ux)
        vxsqrt = np.clip(vx, a_min=0, a_max=None) ** 0.5
        vy = cov_norm * (uyy - uy * uy)
        vysqrt = np.clip(vy, a_min=0, a_max=None) ** 0.5
        vxy = cov_norm * (uxy - ux * uy)

        R = data_range
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2
        C3 = C2 / 2

        L = np.clip((2 * ux * uy + C1) / (ux * ux + uy * uy + C1), a_min=0, a_max=None)
        C = np.clip((2 * vxsqrt * vysqrt + C2) / (vx + vy + C2), a_min=0, a_max=None)
        S = np.clip((vxy + C3) / (vxsqrt * vysqrt + C3), a_min=0, a_max=None)

        result = (L ** luminance_weight) * (C ** contrast_weight) * (S ** structure_weight)
        pad = (win_size - 1) // 2
        mssim = crop(result, pad).mean(dtype=np.float64)
        if full:
            return mssim, result
        return mssim

    def ms_ssim(self, gt: np.ndarray, pred: np.ndarray,
                mask: Optional[np.ndarray] = None,
                scale_weights: Optional[np.ndarray] = None):
        """Masked 5-scale MS-SSIM. Returns (full_image_msssim, masked_msssim)."""
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            mask = np.where(mask > 0, 1.0, 0.0)
            gt = np.where(mask == 0, min(self.dynamic_range), gt)
            pred = np.where(mask == 0, min(self.dynamic_range), pred)

        # Shift to [0, R] for non-negative inputs (matches reference behavior).
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]

        # Wang et al. 2003, Asilomar — see Eq. 7.
        scale_weights = (np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
                         if scale_weights is None else scale_weights)
        # NOTE: reference indexing oddity — when scale_weights is None this
        # array has length 6 with luminance_weights[4]=0; the loop only
        # reaches index `levels-1=4`, so the trailing 0.1333 is never read.
        # Preserved verbatim for leaderboard parity.
        luminance_weights = (np.array([0, 0, 0, 0, 0, 0.1333])
                             if scale_weights is None else scale_weights)
        levels = len(scale_weights)

        downsample_filter = np.ones((2, 2, 2)) / 8
        gtx, gty, gtz = gt.shape

        # MS-SSIM downsamples log2(levels) times — every dim must be ≥ 97.
        target_size = 97
        pad_values = [
            (np.clip((target_size - dim) // 2, a_min=0, a_max=None),
             np.clip(target_size - dim - (target_size - dim) // 2, a_min=0, a_max=None))
            for dim in [gtx, gty, gtz]
        ]
        gt = np.pad(gt, pad_values, mode="edge")
        pred = np.pad(pred, pad_values, mode="edge")
        mask = np.pad(mask, pad_values, mode="edge")

        ms_ssim_vals, ms_ssim_maps = [], []
        for level in range(levels):
            ssim_value_full, ssim_map = self.structural_similarity_at_scale(
                gt, pred,
                luminance_weight=luminance_weights[level],
                contrast_weight=scale_weights[level],
                structure_weight=scale_weights[level],
                data_range=dynamic_range, full=True,
            )
            pad = 3
            ssim_value_masked = (
                crop(ssim_map, pad)[crop(mask, pad).astype(bool)].mean(dtype=np.float64)
            )
            ms_ssim_vals.append(ssim_value_full)
            ms_ssim_maps.append(ssim_value_masked)

            filtered = [fftconvolve(im, downsample_filter, mode="same") for im in [gt, pred]]
            gt, pred, mask = [x[::2, ::2, ::2] for x in [*filtered, mask]]

        ms_ssim_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_vals])
        ms_ssim_mask_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_maps])
        return float(ms_ssim_val), float(ms_ssim_mask_val)


if __name__ == "__main__":
    # Sanity demo: score CT against itself on the first manifest patient.
    # Self-match should give MAE=0, PSNR=inf, MS-SSIM=1.0.
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--patient-id", default=None,
                        help="If omitted, use the first row of the manifest.")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.patient_id is not None:
        df = df[df["patient_id"] == args.patient_id]
    row = df.iloc[0]

    metrics = ImageMetrics()

    # Convert preprocessed [-1, 1] CT back to HU before scoring.
    from utils.hu import to_hu
    ct_norm = sitk.GetArrayFromImage(sitk.ReadImage(row["ct_path"])).astype(np.float32)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(row["mask_path"])).astype(np.float32)
    ct_hu = to_hu(ct_norm)

    print(f"Self-match {row['patient_id']} ({row['region']}):")
    print(metrics.score_patient(ct_hu, ct_hu, mask))
