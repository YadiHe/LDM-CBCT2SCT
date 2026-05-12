from contextlib import nullcontext

import torch
import torch.nn.functional as F

from utils.hu import CLIP_MAX, CLIP_MIN


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * mask).sum() / denom


def full_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def hu_to_norm(hu: float) -> float:
    return ((float(hu) - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)) * 2.0 - 1.0


def high_density_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold_hu: float = 300.0,
) -> torch.Tensor:
    threshold = hu_to_norm(threshold_hu)
    high_mask = mask.to(dtype=pred.dtype) * (target >= threshold).to(dtype=pred.dtype)
    denom = high_mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * high_mask).sum() / denom


def image_gradients(x: torch.Tensor):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def masked_gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_dx, pred_dy = image_gradients(pred)
    tgt_dx, tgt_dy = image_gradients(target)
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    loss_x = ((pred_dx - tgt_dx).abs() * mask_x).sum() / mask_x.sum().clamp_min(1.0)
    loss_y = ((pred_dy - tgt_dy).abs() * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def _gaussian_kernel(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum().clamp_min(1e-12)
    kernel_2d = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
    return kernel_2d.expand(channels, 1, window_size, window_size)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (x * mask).sum(dim=(1, 2, 3)) / denom


def _ssim_maps(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
) -> tuple:
    channels = x.shape[1]
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs_map = (2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    ssim_map = ((2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs_map
    return ssim_map, cs_map


def masked_ms_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    levels: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable 2D MS-SSIM loss averaged over foreground mask pixels."""
    autocast_ctx = torch.cuda.amp.autocast(enabled=False) if pred.is_cuda else nullcontext()
    with autocast_ctx:
        x = (pred.float() + 1.0) * 0.5
        y = (target.float() + 1.0) * 0.5
        m = mask.float().clamp(0.0, 1.0)
        weights = x.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:levels]
        window = _gaussian_kernel(x.shape[1], x.device, x.dtype)

        cs_vals = []
        ssim_val = None
        for level in range(levels):
            ssim_map, cs_map = _ssim_maps(x, y, window)
            ssim_val = _masked_mean(ssim_map, m).clamp(eps, 1.0)
            cs_vals.append(_masked_mean(cs_map, m).clamp(eps, 1.0))

            if level < levels - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)
                m = F.avg_pool2d(m, kernel_size=2, stride=2)

        score = torch.ones_like(ssim_val)
        for level in range(levels - 1):
            score = score * (cs_vals[level] ** weights[level])
        score = score * (ssim_val ** weights[levels - 1])
        return 1.0 - score.mean()


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    grad_weight: float = 0.05,
    full_mae_weight: float = 0.0,
    ms_ssim_weight: float = 0.0,
    high_density_weight: float = 0.0,
    high_density_threshold_hu: float = 300.0,
):
    l1 = masked_l1(pred, target, mask)
    grad = masked_gradient_l1(pred, target, mask) if grad_weight > 0 else pred.new_zeros(())
    full_mae = full_l1(pred, target) if full_mae_weight > 0 else pred.new_zeros(())
    ms_ssim = masked_ms_ssim_loss(pred, target, mask) if ms_ssim_weight > 0 else pred.new_zeros(())
    high_density = (
        high_density_l1(pred, target, mask, threshold_hu=high_density_threshold_hu)
        if high_density_weight > 0 else pred.new_zeros(())
    )
    total = (
        l1
        + float(grad_weight) * grad
        + float(full_mae_weight) * full_mae
        + float(ms_ssim_weight) * ms_ssim
        + float(high_density_weight) * high_density
    )
    return total, {
        "loss/l1": float(l1.detach()),
        "loss/grad": float(grad.detach()),
        "loss/full_mae": float(full_mae.detach()),
        "loss/ms_ssim": float(ms_ssim.detach()),
        "loss/high_density": float(high_density.detach()),
    }
