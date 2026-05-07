import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler

from models.diffusion import Diffusion
from models.blocks import ControlNetPACAUpBlock, nonlinearity, Normalize, TimestepEmbedding, DownBlock, MiddleBlock, ConditionalUpBlock, UpBlock, PACALayer, ZeroConv2d
from models.degradationRemoval import degradation_loss
from models.unet import noise_loss
from utils.losses import PerceptualLoss, SsimLoss
from torch.optim import AdamW

CLIP_MIN = -1024.0
CLIP_MAX = 1500.0
REGION_NAMES = {0: "BB", 1: "AB", 2: "HN", 3: "TH"}


class LatentControlAdapter(nn.Module):
    """Map CBCT latent (3ch) to ControlNet condition channels."""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.proj = ZeroConv2d(in_channels, out_channels)

    def forward(self, x):
        return self.proj(x)


class ModelEMA:
    """Track EMA weights for one or more modules."""

    def __init__(self, modules, decay=0.999):
        self.decay = float(decay)
        if isinstance(modules, nn.Module):
            modules = {"module": modules}
        self.modules = dict(modules)
        self.shadow = {name: self._clone_state(module) for name, module in self.modules.items()}
        self.backup = None

    @staticmethod
    def _clone_state(module):
        return {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

    def update(self):
        for name, module in self.modules.items():
            state = module.state_dict()
            for key, value in state.items():
                value = value.detach()
                if torch.is_floating_point(value):
                    self.shadow[name][key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
                else:
                    self.shadow[name][key] = value.clone()

    def store(self):
        self.backup = {name: self._clone_state(module) for name, module in self.modules.items()}

    def copy_to(self):
        for name, module in self.modules.items():
            module.load_state_dict(self.shadow[name], strict=True)

    def restore(self):
        if self.backup is None:
            return
        for name, module in self.modules.items():
            module.load_state_dict(self.backup[name], strict=True)
        self.backup = None

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay = float(state_dict.get("decay", self.decay))
        shadow = state_dict["shadow"]
        first_value = next(iter(shadow.values())) if shadow else {}
        if isinstance(first_value, torch.Tensor):
            target_name = "unet" if "unet" in self.modules else next(iter(self.modules.keys()))
            self.shadow[target_name] = {key: value.detach().clone() for key, value in shadow.items()}
            return
        self.shadow = {
            name: {key: value.detach().clone() for key, value in module_state.items()}
            for name, module_state in shadow.items()
        }


def _build_lr_scheduler(optimizer, lr_schedule, warmup_steps, total_steps=None, min_lr_ratio=0.1):
    if lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    if lr_schedule == "sd-warmup-constant":
        warmup_steps = max(int(warmup_steps), 0)
        if warmup_steps <= 0:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
        )

    if lr_schedule == "sd-warmup-cosine":
        warmup_steps = max(int(warmup_steps), 0)
        if total_steps is None or total_steps <= warmup_steps:
            raise ValueError("sd-warmup-cosine requires total_steps > warmup_steps")
        decay_steps = max(total_steps - warmup_steps, 1)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(warmup_steps, 1))
            progress = float(step - warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError("lr_schedule must be one of: sd-warmup-constant, sd-warmup-cosine, constant")


def _autocast_dtype(amp_dtype):
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be one of: fp16, bf16")


def _compute_latent_stats(vae, train_loader, device, latent_mode, amp_enabled, amp_dtype, max_batches):
    if max_batches <= 0:
        return {}

    stats = {
        "ct_sum": 0.0,
        "ct_sq_sum": 0.0,
        "ct_count": 0,
        "ct_min": float("inf"),
        "ct_max": float("-inf"),
        "cbct_sum": 0.0,
        "cbct_sq_sum": 0.0,
        "cbct_count": 0,
        "cbct_min": float("inf"),
        "cbct_max": float("-inf"),
    }

    with torch.no_grad():
        for i, (ct_img, cbct_img, _, _) in enumerate(train_loader):
            if i >= max_batches:
                break
            ct_img = ct_img.to(device)
            cbct_img = cbct_img.to(device)
            with autocast(enabled=amp_enabled, dtype=_autocast_dtype(amp_dtype)):
                z_ct = _encode_vae(vae, ct_img, latent_mode).detach().float()
                cbct_z = _encode_vae(vae, cbct_img, latent_mode).detach().float()
            for prefix, z in (("ct", z_ct), ("cbct", cbct_z)):
                stats[f"{prefix}_sum"] += float(z.sum())
                stats[f"{prefix}_sq_sum"] += float((z * z).sum())
                stats[f"{prefix}_count"] += int(z.numel())
                stats[f"{prefix}_min"] = min(stats[f"{prefix}_min"], float(z.min()))
                stats[f"{prefix}_max"] = max(stats[f"{prefix}_max"], float(z.max()))

    out = {}
    for prefix in ("ct", "cbct"):
        count = max(stats[f"{prefix}_count"], 1)
        mean = stats[f"{prefix}_sum"] / count
        var = max(stats[f"{prefix}_sq_sum"] / count - mean * mean, 0.0)
        out[f"latent/{prefix}_mean"] = mean
        out[f"latent/{prefix}_std"] = float(var ** 0.5)
        out[f"latent/{prefix}_min"] = stats[f"{prefix}_min"]
        out[f"latent/{prefix}_max"] = stats[f"{prefix}_max"]
    out["latent/stats_batches"] = max_batches
    return out


class UNetConcatControlPACA(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=256, 
                 dropout_rate=0.0):
        super().__init__()
        time_emb_dim = base_channels * 4

        ch1 = base_channels * 1
        ch2 = base_channels * 2
        ch3 = base_channels * 4
        ch4 = base_channels * 4

        attn_res_64 = False
        attn_res_32 = True
        attn_res_16 = True
        attn_res_8 = True

        self.time_embedding = TimestepEmbedding(time_emb_dim)
        self.region_embedding = nn.Embedding(4, time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels*2, ch1, kernel_size=3, padding=1)

        self.down1 = DownBlock(ch1, ch1, time_emb_dim, attn_res_64, dropout_rate)
        self.down2 = DownBlock(ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.down3 = DownBlock(ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.down4 = DownBlock(ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, downsample=False)

        self.middle = MiddleBlock(ch4, time_emb_dim, dropout_rate)

        self.up4 = ControlNetPACAUpBlock(ch4, ch3, ch4, time_emb_dim, attn_res_8, dropout_rate)
        self.up3 = ControlNetPACAUpBlock(ch3, ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.up2 = ControlNetPACAUpBlock(ch2, ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.up1 = ControlNetPACAUpBlock(ch1, ch1, ch1, time_emb_dim, attn_res_64, dropout_rate, upsample=False)

        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, condition, t, down_paca_control_residuals=None, middle_paca_control_residual=None, down_control_residuals=None, middle_control_residual=None, region_id=None, controlnet_fusion="both"):
        control_paca = True if down_paca_control_residuals is not None else False
        extra_control = True if down_control_residuals is not None else False
        use_add = control_paca and controlnet_fusion in ("add", "both")
        use_paca = control_paca and controlnet_fusion in ("paca", "both")

        if control_paca:
            additional_down_res_1_1, additional_down_res_1_2, additional_down_res_2_1, additional_down_res_2_2, additional_down_res_3_1, additional_down_res_3_2, additional_down_res_4_1, additional_down_res_4_2 = down_paca_control_residuals

        if extra_control:
            extra_additional_down_res_1_1, extra_additional_down_res_1_2, extra_additional_down_res_2_1, extra_additional_down_res_2_2, extra_additional_down_res_3_1, extra_additional_down_res_3_2, extra_additional_down_res_4_1, extra_additional_down_res_4_2 = down_control_residuals

        t_emb = self.time_embedding(t)
        if region_id is not None:
            t_emb = t_emb + self.region_embedding(region_id)
        x = torch.cat((x, condition), dim=1)
        h = self.init_conv(x)         

        h, (down_res_1_1, down_res_1_2) = self.down1(h, t_emb)
        h, (down_res_2_1, down_res_2_2) = self.down2(h, t_emb)
        h, (down_res_3_1, down_res_3_2) = self.down3(h, t_emb)
        h, (down_res_4_1, down_res_4_2) = self.down4(h, t_emb)

        if use_add:
            down_res_1_1 = down_res_1_1 + additional_down_res_1_1
            down_res_1_2 = down_res_1_2 + additional_down_res_1_2
            down_res_2_1 = down_res_2_1 + additional_down_res_2_1
            down_res_2_2 = down_res_2_2 + additional_down_res_2_2
            down_res_3_1 = down_res_3_1 + additional_down_res_3_1
            down_res_3_2 = down_res_3_2 + additional_down_res_3_2
            down_res_4_1 = down_res_4_1 + additional_down_res_4_1
            down_res_4_2 = down_res_4_2 + additional_down_res_4_2

        if extra_control:
            down_res_1_1 = down_res_1_1 + extra_additional_down_res_1_1
            down_res_1_2 = down_res_1_2 + extra_additional_down_res_1_2
            down_res_2_1 = down_res_2_1 + extra_additional_down_res_2_1
            down_res_2_2 = down_res_2_2 + extra_additional_down_res_2_2
            down_res_3_1 = down_res_3_1 + extra_additional_down_res_3_1
            down_res_3_2 = down_res_3_2 + extra_additional_down_res_3_2
            down_res_4_1 = down_res_4_1 + extra_additional_down_res_4_1
            down_res_4_2 = down_res_4_2 + extra_additional_down_res_4_2

        h = self.middle(h, t_emb)

        paca_4 = (additional_down_res_4_1, additional_down_res_4_2) if use_paca else None
        paca_3 = (additional_down_res_3_1, additional_down_res_3_2) if use_paca else None
        paca_2 = (additional_down_res_2_1, additional_down_res_2_2) if use_paca else None
        paca_1 = (additional_down_res_1_1, additional_down_res_1_2) if use_paca else None

        if control_paca and use_add:
            h = h + middle_paca_control_residual
        if extra_control:
            h = h + middle_control_residual
        if control_paca:
            h = self.up4(h, (down_res_4_1, down_res_4_2), paca_4, t_emb)
            h = self.up3(h, (down_res_3_1, down_res_3_2), paca_3, t_emb)
            h = self.up2(h, (down_res_2_1, down_res_2_2), paca_2, t_emb)
            h = self.up1(h, (down_res_1_1, down_res_1_2), paca_1, t_emb)
        
        if not control_paca:
            h = self.up4(h, (down_res_4_1, down_res_4_2), None, t_emb)
            h = self.up3(h, (down_res_3_1, down_res_3_2), None, t_emb)
            h = self.up2(h, (down_res_2_1, down_res_2_2), None, t_emb)
            h = self.up1(h, (down_res_1_1, down_res_1_2), None, t_emb)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h
    
def load_unet_concat_control_paca(unet_save_path=None, paca_save_path=None, unet_trainable=False, paca_trainable=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unetConcatControlPACA = UNetConcatControlPACA().to(device)

    if paca_save_path:
        paca_state_dict = torch.load(paca_save_path, map_location=device)
        _, paca_unexpected_keys = unetConcatControlPACA.load_state_dict(paca_state_dict, strict=False)
        if paca_unexpected_keys:
            print(f"Unexpected keys in PACA state_dict: {paca_unexpected_keys}")

    unet_state_dict = None
    if unet_save_path is None:
        print("UNet initialized with random weights.")
    else: 
        unet_state_dict = torch.load(unet_save_path, map_location=device)
        _, unetControlPACA_unexpected_keys = unetConcatControlPACA.load_state_dict(unet_state_dict, strict=False)
        if unetControlPACA_unexpected_keys:
            print(f"Unexpected keys in UNetControlPACA state_dict: {unetControlPACA_unexpected_keys}")

    for param in unetConcatControlPACA.parameters():
        param.requires_grad = unet_trainable
    
    paca_params = 0
    for name, param in unetConcatControlPACA.named_parameters():
        if 'paca' in name.lower():
            param.requires_grad = paca_trainable
            paca_params += param.numel()
    
    if unet_save_path:
        unet_control_paca_params = sum(p.numel() for p in unetConcatControlPACA.parameters())
        unet_params = sum(p.numel() for p in unet_state_dict.values())
        if unet_control_paca_params - paca_params != unet_params:
            print(f"WARNING: UNetControlPACA parameters - PACA parameters should be equal to the loaded state_dict parameters.")
            print(f"Loaded state_dict parameters: {unet_params}")
            print(f"UNetControlPACA parameters: {unet_control_paca_params}")
            print(f"UNetControlPACA parameters - PACA parameters: {unet_control_paca_params - paca_params}")
        print(f"UNetControlPACA loaded from {unet_save_path}")

    return unetConcatControlPACA

def _to_hu(x_norm):
    return ((x_norm + 1.0) * 0.5 * (CLIP_MAX - CLIP_MIN)) + CLIP_MIN


def _to_image01(x_norm):
    return ((x_norm + 1.0) * 0.5).clamp(0.0, 1.0)


def _add_panel_label(image, label):
    out = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image.copy()
    out[:18, :, :] = 0.0
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray((out * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        draw.text((4, 3), label, fill=(255, 255, 255))
        out = np.asarray(pil).astype(np.float32) / 255.0
    except Exception:
        label_width = min(len(label) * 6 + 8, out.shape[1])
        out[2:16, 2:label_width, :] = 1.0
    return out


def _make_comparison_panel(cbct, ct, sct, error):
    panels = [
        _add_panel_label(cbct, "CBCT"),
        _add_panel_label(ct, "CT"),
        _add_panel_label(sct, "sCT"),
        _add_panel_label(error, "|err|"),
    ]
    separator = np.ones((panels[0].shape[0], 3, 3), dtype=panels[0].dtype)
    return np.concatenate([panels[0], separator, panels[1], separator, panels[2], separator, panels[3]], axis=1)


def _encode_vae(vae, x, latent_mode="mu"):
    mu, logvar = vae.encode(x)
    if latent_mode == "sample":
        return vae.reparameterize(mu, logvar)
    return mu


def _masked_psnr(pred, target, mask):
    mse = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    return float(20.0 * torch.log10(torch.tensor(2.0, device=pred.device)) - 10.0 * torch.log10(mse.clamp_min(1e-12)))


def _masked_mse_loss(pred, target, mask):
    mask = mask.to(dtype=pred.dtype)
    denom = (mask.sum() * pred.size(1)).clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def _masked_l1_loss(pred, target, mask):
    mask = mask.to(dtype=pred.dtype)
    denom = (mask.sum() * pred.size(1)).clamp_min(1.0)
    return ((pred - target).abs() * mask).sum() / denom


def _masked_mse_loss_per_sample(pred, target, mask):
    mask = mask.to(dtype=pred.dtype)
    sq = ((pred - target) ** 2) * mask
    per_denom = (mask.sum(dim=[1, 2, 3]) * pred.size(1)).clamp_min(1.0)
    return sq.sum(dim=[1, 2, 3]) / per_denom


def _masked_l1_loss_per_sample(pred, target, mask):
    mask = mask.to(dtype=pred.dtype)
    abs_diff = (pred - target).abs() * mask
    per_denom = (mask.sum(dim=[1, 2, 3]) * pred.size(1)).clamp_min(1.0)
    return abs_diff.sum(dim=[1, 2, 3]) / per_denom


def _min_snr_weights(t, alpha_cumprod, gamma=5.0):
    """Min-SNR-γ weights for epsilon-prediction L1/L2 loss.
    Reference: Hang et al. 2023, "Efficient Diffusion Training via Min-SNR Weighting Strategy".
    Per-sample weight: w(t) = min(SNR_t, gamma) / SNR_t,  SNR_t = a_t / (1 - a_t)
    """
    a = alpha_cumprod[t].clamp_min(1e-8)
    snr = a / (1.0 - a).clamp_min(1e-8)
    w = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    return w


def _masked_ssim(pred, target, mask):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")
    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    mask_np = mask.detach().float().cpu().numpy()
    values = []
    for p, t, m in zip(pred_np, target_np, mask_np):
        p2 = p.squeeze()
        t2 = t.squeeze()
        m2 = m.squeeze() > 0.5
        # Keep padding identical so SSIM is not rewarded or punished by background.
        p2 = np.where(m2, p2, -1.0)
        t2 = np.where(m2, t2, -1.0)
        values.append(structural_similarity(t2, p2, data_range=2.0, win_size=11))
    return float(np.mean(values)) if values else float("nan")


def _control_inputs(cbct_img, cbct_z, dr_module, control_adapter, use_controlnet, use_dr, control_source):
    if not use_controlnet:
        return None, None
    if control_source == "dr":
        control_feature, intermediate_preds = dr_module(cbct_img)
        return control_feature, intermediate_preds
    if control_source == "cbct_latent":
        return control_adapter(cbct_z), None
    raise ValueError(f"Unknown control_source: {control_source}")


def _predict_noise(unet, controlnet, z_noisy, cbct_z, t, control_feature, region_id, use_controlnet, controlnet_fusion):
    if use_controlnet:
        down_res, middle_res = controlnet(z_noisy, control_feature, t)
        return unet(
            z_noisy,
            cbct_z,
            t,
            down_paca_control_residuals=down_res,
            middle_paca_control_residual=middle_res,
            region_id=region_id,
            controlnet_fusion=controlnet_fusion,
        )
    return unet(z_noisy, cbct_z, t, region_id=region_id)


def _ddim_sample(vae, unet, controlnet, dr_module, control_adapter, diffusion, cbct_img, region_id,
                 use_controlnet, use_dr, control_source, controlnet_fusion, latent_mode, steps, amp_enabled,
                 amp_dtype="fp16", sampler_init="noise", sampler_t_start=999, sampler_alpha=1.0):
    cbct_z = _encode_vae(vae, cbct_img, latent_mode)
    control_feature, _ = _control_inputs(cbct_img, cbct_z, dr_module, control_adapter, use_controlnet, use_dr, control_source)
    generator = torch.Generator(device=cbct_img.device).manual_seed(0)
    sampler_t_start = int(min(max(sampler_t_start, 0), diffusion.timesteps - 1))
    timesteps = torch.linspace(sampler_t_start, 0, steps, device=cbct_z.device).long()
    noise = torch.randn(cbct_z.shape, device=cbct_z.device, dtype=cbct_z.dtype, generator=generator)
    if sampler_init == "cbct":
        alpha_eff = (sampler_alpha * diffusion.alpha_cumprod[timesteps[0]]).clamp(0.0, 1.0).view(1, 1, 1, 1)
        x = torch.sqrt(alpha_eff) * cbct_z + torch.sqrt(1.0 - alpha_eff) * noise
    elif sampler_init == "noise":
        x = noise
    else:
        raise ValueError("sampler_init must be one of: noise, cbct")

    for idx, t_scalar in enumerate(timesteps):
        t = torch.full((cbct_z.size(0),), int(t_scalar.item()), device=cbct_z.device, dtype=torch.long)
        with autocast(enabled=amp_enabled, dtype=_autocast_dtype(amp_dtype)):
            eps = _predict_noise(unet, controlnet, x, cbct_z, t, control_feature, region_id, use_controlnet, controlnet_fusion)
        alpha_t = diffusion.alpha_cumprod[t_scalar].view(1, 1, 1, 1)
        pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t)
        if idx == len(timesteps) - 1:
            x = pred_x0
        else:
            prev_t = timesteps[idx + 1]
            alpha_prev = diffusion.alpha_cumprod[prev_t].view(1, 1, 1, 1)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * eps
    return vae.decode(x).clamp(-1.0, 1.0)


def _eval_decoded_metrics(vae, unet, controlnet, dr_module, control_adapter, diffusion, val_loader, device,
                          use_controlnet, use_dr, control_source, controlnet_fusion, latent_mode, ddim_steps,
                          amp_enabled, amp_dtype="fp16", max_val_batches=None, sampler_init="noise", sampler_t_start=999,
                          sampler_alpha=1.0):
    totals = {"mae_hu": 0.0, "psnr": 0.0, "ssim": 0.0}
    region_totals = {name: [0.0, 0] for name in REGION_NAMES.values()}
    count = 0
    with torch.no_grad():
        for i, (ct_img, cbct_img, mask, region_id) in enumerate(val_loader):
            if max_val_batches is not None and i >= max_val_batches:
                break
            ct_img = ct_img.to(device)
            cbct_img = cbct_img.to(device)
            mask = mask.to(device)
            region_id = region_id.to(device)
            sct = _ddim_sample(vae, unet, controlnet, dr_module, control_adapter, diffusion, cbct_img, region_id,
                               use_controlnet, use_dr, control_source, controlnet_fusion, latent_mode, ddim_steps,
                               amp_enabled, amp_dtype, sampler_init, sampler_t_start, sampler_alpha)
            sct = sct * mask + (-1.0) * (1.0 - mask)
            mae_per = ((_to_hu(sct) - _to_hu(ct_img)).abs() * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0)
            psnr = _masked_psnr(sct, ct_img, mask)
            ssim = _masked_ssim(sct, ct_img, mask)
            batch_n = ct_img.size(0)
            totals["mae_hu"] += float(mae_per.mean()) * batch_n
            totals["psnr"] += psnr * batch_n
            totals["ssim"] += ssim * batch_n
            count += batch_n
            for mae, rid in zip(mae_per.detach().cpu().tolist(), region_id.detach().cpu().tolist()):
                name = REGION_NAMES[int(rid)]
                region_totals[name][0] += float(mae)
                region_totals[name][1] += 1

    metrics = {f"val/{k}": v / max(count, 1) for k, v in totals.items()}
    for name, (total, n) in region_totals.items():
        if n:
            metrics[f"val/mae_hu_{name}"] = total / n
    return metrics


def _log_fixed_val_images(vae, unet, controlnet, dr_module, control_adapter, diffusion, fixed_val_batch, device,
                          use_controlnet, use_dr, control_source, controlnet_fusion, latent_mode, ddim_steps,
                          amp_enabled, amp_dtype, wandb_logger, epoch, max_images=8, sampler_init="noise",
                          sampler_t_start=999, sampler_alpha=1.0):
    if not wandb_logger or fixed_val_batch is None:
        return
    ct_img, cbct_img, mask, region_id, captions = fixed_val_batch
    total_images = min(int(max_images), ct_img.size(0))
    chunk_size = min(8, max(total_images, 1))
    for start in range(0, total_images, chunk_size):
        end = min(start + chunk_size, total_images)
        ct_chunk = ct_img[start:end].to(device)
        cbct_chunk = cbct_img[start:end].to(device)
        mask_chunk = mask[start:end].to(device)
        region_chunk = region_id[start:end].to(device)
        with torch.no_grad():
            sct = _ddim_sample(vae, unet, controlnet, dr_module, control_adapter, diffusion, cbct_chunk, region_chunk,
                               use_controlnet, use_dr, control_source, controlnet_fusion, latent_mode, ddim_steps,
                               amp_enabled, amp_dtype, sampler_init, sampler_t_start, sampler_alpha)
        sct = sct * mask_chunk + (-1.0) * (1.0 - mask_chunk)
        err = ((_to_hu(sct) - _to_hu(ct_chunk)).abs() * mask_chunk).clamp(0, 300) / 300.0
        for local_i in range(ct_chunk.size(0)):
            i = start + local_i
            caption = captions[i] if captions else f"sample_{i}"
            panel = _make_comparison_panel(
                _to_image01(cbct_chunk[local_i]).squeeze().cpu().numpy(),
                _to_image01(ct_chunk[local_i]).squeeze().cpu().numpy(),
                _to_image01(sct[local_i]).squeeze().cpu().numpy(),
                err[local_i].squeeze().cpu().numpy(),
            )
            wandb_logger.log_image(f"fixed_val/comparison_{i}", panel, caption, step=epoch)
        del ct_chunk, cbct_chunk, mask_chunk, region_chunk, sct, err


def train_unet_concat_control_paca(
    vae,
    unet,
    controlnet,
    dr_module,
    train_loader,
    val_loader,
    epochs=1000,
    save_dir='.',
    predict_dir="predictions",
    early_stopping=None,
    gamma=1.0,
    learning_rate=1e-5,
    weight_decay=1e-4,
    lr_schedule="sd-warmup-constant",
    warmup_steps=1000,
    wandb_logger=None,
    max_train_batches=None,
    max_val_batches=None,
    grad_accum_steps=1,
    amp_enabled=None,
    amp_dtype="fp16",
    control_adapter=None,
    use_dr=False,
    use_controlnet=False,
    control_source="cbct_latent",
    controlnet_fusion="add",
    latent_mode="mu",
    ddim_steps=50,
    sampler_init="noise",
    sampler_t_start=999,
    sampler_alpha=1.0,
    use_ema=False,
    ema_decay=0.999,
    ema_path=None,
    latent_stats_batches=4,
    eval_every=10,
    fixed_val_batch=None,
    fixed_val_max_images=48,
    loss_type="l1",
    use_min_snr_weight=False,
    min_snr_gamma=5.0,
    cosine_min_lr_ratio=0.1,
):
    if loss_type not in ("mse", "l1"):
        raise ValueError("loss_type must be 'mse' or 'l1'")
    _per_sample_diff_loss_fn = _masked_l1_loss_per_sample if loss_type == "l1" else _masked_mse_loss_per_sample
    if controlnet_fusion not in ("add", "paca", "both"):
        raise ValueError("controlnet_fusion must be one of: add, paca, both")
    if latent_mode not in ("mu", "sample"):
        raise ValueError("latent_mode must be one of: mu, sample")
    if sampler_init not in ("noise", "cbct"):
        raise ValueError("sampler_init must be one of: noise, cbct")
    if lr_schedule not in ("sd-warmup-constant", "sd-warmup-cosine", "constant"):
        raise ValueError("lr_schedule must be one of: sd-warmup-constant, constant")
    if use_dr and (not use_controlnet or control_source != "dr"):
        raise ValueError("--use-dr is only supported with --use-controlnet --control-source dr")
    if use_controlnet and control_source == "dr" and dr_module is None:
        raise ValueError("control-source=dr requires a DR module")
    if use_controlnet and control_source == "cbct_latent" and control_adapter is None:
        raise ValueError("control-source=cbct_latent requires a LatentControlAdapter")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)
    if amp_enabled is None:
        amp_enabled = torch.cuda.is_available()
    if amp_dtype not in ("fp16", "bf16"):
        raise ValueError("amp_dtype must be one of: fp16, bf16")
    if amp_enabled and amp_dtype == "bf16" and torch.cuda.is_available():
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not bf16_supported:
            raise RuntimeError("CUDA bf16 AMP requested but torch.cuda.is_bf16_supported() is false")
    grad_accum_steps = max(int(grad_accum_steps), 1)

    vae.to(device).eval()
    unet.to(device)
    if controlnet is not None:
        controlnet.to(device)
    if dr_module is not None:
        dr_module.to(device)
    if control_adapter is not None:
        control_adapter.to(device)

    unet_params = [p for p in unet.parameters() if p.requires_grad]
    controlnet_params = [p for p in controlnet.parameters() if use_controlnet and p.requires_grad] if controlnet is not None else []
    dr_params = [p for p in dr_module.parameters() if use_dr and p.requires_grad] if dr_module is not None else []
    adapter_params = [p for p in control_adapter.parameters() if use_controlnet and control_source == "cbct_latent" and p.requires_grad] if control_adapter is not None else []
    params_to_train = unet_params + controlnet_params + dr_params + adapter_params
    if not params_to_train:
        raise ValueError("No trainable parameters selected.")

    print(f"AMP Enabled: {amp_enabled} ({amp_dtype if amp_enabled else 'off'})")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"LR schedule: {lr_schedule} (warmup_steps={warmup_steps})")
    print(f"Modules: use_dr={use_dr}, use_controlnet={use_controlnet}, control_source={control_source}, fusion={controlnet_fusion}")
    print(f"Trainable parameters UNet={sum(p.numel() for p in unet_params)} ControlNet={sum(p.numel() for p in controlnet_params)} DR={sum(p.numel() for p in dr_params)} Adapter={sum(p.numel() for p in adapter_params)}")

    train_batches_for_schedule = min(len(train_loader), max_train_batches) if max_train_batches is not None else len(train_loader)
    optimizer_steps_per_epoch = (train_batches_for_schedule + grad_accum_steps - 1) // grad_accum_steps
    total_optimizer_steps = max(epochs * optimizer_steps_per_epoch, 1)
    warmup_ratio = float(warmup_steps) / float(total_optimizer_steps)
    print(
        f"Schedule steps: train_batches={train_batches_for_schedule}, "
        f"optimizer_steps/epoch={optimizer_steps_per_epoch}, total_steps={total_optimizer_steps}, "
        f"warmup_ratio={warmup_ratio:.3f}"
    )

    latent_stats = _compute_latent_stats(
        vae, train_loader, device, latent_mode, amp_enabled, amp_dtype, max(int(latent_stats_batches), 0)
    )
    if latent_stats:
        print(
            "Latent stats "
            f"ct mean={latent_stats['latent/ct_mean']:.4f} std={latent_stats['latent/ct_std']:.4f} "
            f"min={latent_stats['latent/ct_min']:.4f} max={latent_stats['latent/ct_max']:.4f} | "
            f"cbct mean={latent_stats['latent/cbct_mean']:.4f} std={latent_stats['latent/cbct_std']:.4f} "
            f"min={latent_stats['latent/cbct_min']:.4f} max={latent_stats['latent/cbct_max']:.4f}"
        )

    optimizer = AdamW(params_to_train, lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == "fp16")
    scheduler = _build_lr_scheduler(optimizer, lr_schedule, warmup_steps,
                                    total_steps=total_optimizer_steps,
                                    min_lr_ratio=cosine_min_lr_ratio)
    ema_modules = {"unet": unet}
    if controlnet is not None and use_controlnet:
        ema_modules["controlnet"] = controlnet
    if control_adapter is not None and use_controlnet and control_source == "cbct_latent":
        ema_modules["control_adapter"] = control_adapter
    if dr_module is not None and use_dr:
        ema_modules["dr_module"] = dr_module
    ema = ModelEMA(ema_modules, decay=ema_decay) if use_ema else None
    if ema is not None and ema_path:
        ema.load_state_dict(torch.load(ema_path, map_location=device))
        print(f"EMA resumed from: {ema_path}")
    if ema is not None:
        print(f"EMA enabled: decay={ema.decay}, modules={','.join(ema.modules.keys())}")
    diffusion = Diffusion(device, timesteps=1000)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    global_step = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        unet.train()
        if controlnet is not None:
            controlnet.train(use_controlnet)
        if dr_module is not None:
            dr_module.train(use_dr)
        if control_adapter is not None:
            control_adapter.train(use_controlnet and control_source == "cbct_latent")

        train_loss_total = train_loss_diff = train_loss_dr = 0.0
        train_batches = min(len(train_loader), max_train_batches) if max_train_batches is not None else len(train_loader)
        train_valid_batches = 0
        train_skipped_nonfinite = 0
        train_skipped_overflow = 0
        optimizer_steps = 0
        accum_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for i, (ct_img, cbct_img, mask, region_id) in enumerate(train_loader):
            if max_train_batches is not None and i >= max_train_batches:
                break
            ct_img = ct_img.to(device)
            cbct_img = cbct_img.to(device)
            mask = mask.to(device)
            region_id = region_id.to(device)

            with torch.no_grad(), autocast(enabled=amp_enabled, dtype=_autocast_dtype(amp_dtype)):
                z_ct = _encode_vae(vae, ct_img, latent_mode)
                cbct_z = _encode_vae(vae, cbct_img, latent_mode)

            with autocast(enabled=amp_enabled, dtype=_autocast_dtype(amp_dtype)):
                control_feature, intermediate_preds = _control_inputs(
                    cbct_img, cbct_z, dr_module, control_adapter, use_controlnet, use_dr, control_source
                )
                t = diffusion.sample_timesteps(z_ct.size(0))
                noise = torch.randn_like(z_ct)
                z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)
                pred_noise = _predict_noise(unet, controlnet, z_noisy_ct, cbct_z, t, control_feature, region_id,
                                            use_controlnet, controlnet_fusion)
                latent_mask = F.avg_pool2d(mask.float(), kernel_size=4, stride=4) > 0.5
                per_sample_loss = _per_sample_diff_loss_fn(pred_noise, noise, latent_mask)
                if use_min_snr_weight:
                    snr_w = _min_snr_weights(t, diffusion.alpha_cumprod, gamma=min_snr_gamma).to(per_sample_loss.dtype)
                    loss_diff = (per_sample_loss * snr_w).mean()
                else:
                    loss_diff = per_sample_loss.mean()
                loss_dr = degradation_loss(intermediate_preds, ct_img, mask) if use_dr else torch.zeros((), device=device)
                total_loss = loss_diff + gamma * loss_dr
                scaled_loss = total_loss / grad_accum_steps

            if not torch.isfinite(total_loss.detach()):
                train_skipped_nonfinite += 1
                accum_batches = 0
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"Warning: skipped non-finite training loss at epoch {epoch+1}, "
                    f"batch {i+1}: total={float(total_loss.detach())}"
                )
                continue

            scaler.scale(scaled_loss).backward()
            accum_batches += 1
            should_step = (accum_batches >= grad_accum_steps) or ((i + 1) == train_batches)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
                scale_before = scaler.get_scale()
                if not torch.isfinite(grad_norm):
                    train_skipped_overflow += 1
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accum_batches = 0
                    print(
                        f"Warning: skipped optimizer bookkeeping after non-finite grad norm "
                        f"at epoch {epoch+1}, batch {i+1}: grad_norm={float(grad_norm.detach())}"
                    )
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scale_after = scaler.get_scale()
                amp_overflow = scaler.is_enabled() and scale_after < scale_before
                if amp_overflow:
                    train_skipped_overflow += 1
                    print(
                        f"Warning: skipped scheduler/EMA after AMP overflow at epoch {epoch+1}, "
                        f"batch {i+1}: scale {scale_before:.1f}->{scale_after:.1f}"
                    )
                else:
                    scheduler.step()
                    if ema is not None:
                        ema.update()
                    optimizer_steps += 1
                    global_step += 1
                accum_batches = 0

            train_loss_total += float(total_loss.detach())
            train_loss_diff += float(loss_diff.detach())
            train_loss_dr += float(loss_dr.detach())
            train_valid_batches += 1

        avg_train_loss_total = train_loss_total / max(train_valid_batches, 1)
        avg_train_loss_diff = train_loss_diff / max(train_valid_batches, 1)
        avg_train_loss_dr = train_loss_dr / max(train_valid_batches, 1)

        if ema is not None:
            ema.store()
            ema.copy_to()

        try:
            unet.eval()
            if controlnet is not None:
                controlnet.eval()
            if dr_module is not None:
                dr_module.eval()
            if control_adapter is not None:
                control_adapter.eval()
            val_loss_total = val_loss_diff = val_loss_dr = 0.0
            val_batches = min(len(val_loader), max_val_batches) if max_val_batches is not None else len(val_loader)
            val_generator = torch.Generator(device=device).manual_seed(42)

            with torch.no_grad():
                for i, (ct_img, cbct_img, mask, region_id) in enumerate(val_loader):
                    if max_val_batches is not None and i >= max_val_batches:
                        break
                    ct_img = ct_img.to(device)
                    cbct_img = cbct_img.to(device)
                    mask = mask.to(device)
                    region_id = region_id.to(device)

                    with autocast(enabled=amp_enabled, dtype=_autocast_dtype(amp_dtype)):
                        z_ct = _encode_vae(vae, ct_img, latent_mode)
                        cbct_z = _encode_vae(vae, cbct_img, latent_mode)
                        control_feature, intermediate_preds = _control_inputs(
                            cbct_img, cbct_z, dr_module, control_adapter, use_controlnet, use_dr, control_source
                        )
                        t = diffusion.sample_timesteps(z_ct.size(0), generator=val_generator)
                        noise = torch.randn(z_ct.shape, device=z_ct.device, dtype=z_ct.dtype, generator=val_generator)
                        z_noisy_ct = diffusion.add_noise(z_ct, t, noise=noise)
                        pred_noise = _predict_noise(unet, controlnet, z_noisy_ct, cbct_z, t, control_feature, region_id,
                                                    use_controlnet, controlnet_fusion)
                        latent_mask = F.avg_pool2d(mask.float(), kernel_size=4, stride=4) > 0.5
                        per_sample_loss = _per_sample_diff_loss_fn(pred_noise, noise, latent_mask)
                        if use_min_snr_weight:
                            snr_w = _min_snr_weights(t, diffusion.alpha_cumprod, gamma=min_snr_gamma).to(per_sample_loss.dtype)
                            loss_diff = (per_sample_loss * snr_w).mean()
                        else:
                            loss_diff = per_sample_loss.mean()
                        loss_dr = degradation_loss(intermediate_preds, ct_img, mask) if use_dr else torch.zeros((), device=device)
                        total_loss = loss_diff + gamma * loss_dr

                    val_loss_total += float(total_loss.detach())
                    val_loss_diff += float(loss_diff.detach())
                    val_loss_dr += float(loss_dr.detach())

            avg_val_loss_total = val_loss_total / max(val_batches, 1)
            avg_val_loss_diff = val_loss_diff / max(val_batches, 1)
            avg_val_loss_dr = val_loss_dr / max(val_batches, 1)

            early_stopping_counter += 1
            epoch_time = time.time() - epoch_start
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            current_lr = optimizer.param_groups[0]['lr']
            step_time_ms = epoch_time * 1000.0 / max(train_batches, 1)

            val_label = "ValEMA" if ema is not None else "Val"
            print(
                f"Epoch {epoch+1} | Train {avg_train_loss_total:.6f} (Diff {avg_train_loss_diff:.6f}, DR {avg_train_loss_dr:.6f}) | "
                f"{val_label} {avg_val_loss_total:.6f} (Diff {avg_val_loss_diff:.6f}, DR {avg_val_loss_dr:.6f}) | "
                f"LR {current_lr:.2e} | {epoch_time:.1f}s | GPU {gpu_mem:.2f}GB"
            )

            extra_metrics = {
                "train/loss_diff": avg_train_loss_diff,
                "train/loss_dr": avg_train_loss_dr,
                "val/loss_diff": avg_val_loss_diff,
                "val/loss_dr": avg_val_loss_dr,
                "train/batches": train_batches,
                "train/valid_batches": train_valid_batches,
                "train/skipped_nonfinite_batches": train_skipped_nonfinite,
                "train/skipped_overflow_steps": train_skipped_overflow,
                "val/batches": val_batches,
                "train/optimizer_steps": optimizer_steps,
                "train/global_step": global_step,
                "train/grad_accum_steps": grad_accum_steps,
                "lr/current": current_lr,
                "lr/warmup_steps": warmup_steps,
                "lr/warmup_ratio": warmup_ratio,
                "lr/total_steps": total_optimizer_steps,
                "gpu_mem_max_gb": gpu_mem,
                "epoch_time_sec": epoch_time,
                "step_time_ms": step_time_ms,
                "trainable_params": sum(p.numel() for p in params_to_train),
                "sampler/t_start": sampler_t_start,
                "sampler/alpha": sampler_alpha,
                "ema/enabled": 1 if ema is not None else 0,
                "ema/decay": ema.decay if ema is not None else 0.0,
                "amp/enabled": 1 if amp_enabled else 0,
                "amp/dtype_fp16": 1 if amp_enabled and amp_dtype == "fp16" else 0,
                "amp/dtype_bf16": 1 if amp_enabled and amp_dtype == "bf16" else 0,
            }
            extra_metrics.update(latent_stats)

            if (epoch + 1) % max(eval_every, 1) == 0:
                decoded_metrics = _eval_decoded_metrics(
                    vae, unet, controlnet, dr_module, control_adapter, diffusion, val_loader, device,
                    use_controlnet, use_dr, control_source, controlnet_fusion, latent_mode, ddim_steps,
                    amp_enabled, amp_dtype, max_val_batches=max_val_batches, sampler_init=sampler_init,
                    sampler_t_start=sampler_t_start, sampler_alpha=sampler_alpha,
                )
                extra_metrics.update(decoded_metrics)
                _log_fixed_val_images(
                    vae, unet, controlnet, dr_module, control_adapter, diffusion, fixed_val_batch, device,
                    use_controlnet, use_dr, control_source, controlnet_fusion, latent_mode, ddim_steps,
                    amp_enabled, amp_dtype, wandb_logger, epoch + 1, max_images=fixed_val_max_images, sampler_init=sampler_init,
                    sampler_t_start=sampler_t_start, sampler_alpha=sampler_alpha,
                )

            if wandb_logger:
                wandb_logger.log_training_step(
                    epoch=epoch + 1,
                    train_loss=avg_train_loss_total,
                    val_loss=avg_val_loss_total,
                    learning_rate=current_lr,
                    extra_metrics=extra_metrics,
                )

            losses_are_finite = np.isfinite(avg_train_loss_total) and np.isfinite(avg_val_loss_total)
            if not losses_are_finite:
                print(
                    f"Warning: non-finite epoch metrics at epoch {epoch+1}; "
                    "skipping best checkpoint save."
                )

            if losses_are_finite and avg_val_loss_total < best_val_loss:
                best_val_loss = avg_val_loss_total
                early_stopping_counter = 0
                if ema is not None:
                    torch.save(ema.backup["unet"], os.path.join(save_dir, "unet_full.pth"))
                    torch.save(unet.state_dict(), os.path.join(save_dir, "unet_ema.pth"))
                    torch.save(ema.state_dict(), os.path.join(save_dir, "unet_ema_state.pth"))
                    paca_state_source = unet.state_dict()
                else:
                    torch.save(unet.state_dict(), os.path.join(save_dir, "unet_full.pth"))
                    paca_state_source = unet.state_dict()
                if controlnet is not None and use_controlnet:
                    if ema is not None and "controlnet" in ema.backup:
                        torch.save(ema.backup["controlnet"], os.path.join(save_dir, "controlnet_full.pth"))
                    torch.save(controlnet.state_dict(), os.path.join(save_dir, "controlnet.pth"))
                if dr_module is not None and use_dr:
                    if ema is not None and "dr_module" in ema.backup:
                        torch.save(ema.backup["dr_module"], os.path.join(save_dir, "dr_module_full.pth"))
                    torch.save(dr_module.state_dict(), os.path.join(save_dir, "dr_module.pth"))
                if control_adapter is not None and use_controlnet and control_source == "cbct_latent":
                    if ema is not None and "control_adapter" in ema.backup:
                        torch.save(ema.backup["control_adapter"], os.path.join(save_dir, "control_adapter_full.pth"))
                    torch.save(control_adapter.state_dict(), os.path.join(save_dir, "control_adapter.pth"))
                paca_state_dict = {k: v for k, v in paca_state_source.items() if 'paca' in k.lower()}
                if paca_state_dict:
                    torch.save(paca_state_dict, os.path.join(save_dir, "paca_layers.pth"))
                suffix = " using EMA" if ema is not None else ""
                print(f"Saved best epoch {epoch+1}: val {avg_val_loss_total:.6f}{suffix}")
        finally:
            if ema is not None:
                ema.restore()

        if early_stopping and early_stopping_counter >= early_stopping:
            print(f"Early stopped after {early_stopping} epochs with no improvement.")
            break

    print("Training finished.")
