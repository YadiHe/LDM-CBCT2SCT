"""MedVAE adapter — wrap stanfordmimi/MedVAE so it presents the same interface
as our project's `models.vae.VAE` (so D1 / patient_level_vae_metrics can swap
without changes).

**Fully offline**: weights and library source are vendored in-tree:
  - third_party/medvae/                 ← MedVAE library code
  - checkpoints/medvae/medvae_4x3.yaml  ← model config
  - checkpoints/medvae/vae_4x_3c_2D.ckpt ← model weights (214 MB)

No HuggingFace Hub call, no network. To re-pull a fresh copy, see
README or `huggingface-cli download stanfordmimi/MedVAE`.

Key differences from our self-trained VAE handled internally:

  • input channel : MedVAE 4_3_2d expects 3-channel input. We broadcast 1→3 on
                    encode and mean-reduce 3→1 on decode.
  • latent dist   : MedVAE latent has non-trivial per-channel mean / std,
                    measured on v2 train (3000 slices):
                        ch0: mean=+10.87 std=6.58
                        ch1: mean= -9.26 std=5.00
                        ch2: mean= -0.81 std=1.43
                    We z-score per channel so the exposed latent is ~N(0, 1),
                    matching what D1's noise schedule assumes (D1 should use
                    `--latent-scale 1.0`, no further scaling needed).
  • parameters    : all weights frozen (requires_grad=False, eval mode).

API surface (same as `models.vae.VAE`):
  encode(x: (B,1,H,W)) → (mu: (B,3,H/4,W/4), logvar: same)
  reparameterize(mu, logvar) → z
  decode(z: (B,3,h,w)) → x_recon: (B,1,h*4,w*4)
  forward(x) → (z, mu, logvar, recon)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_DIR   = _PROJECT_ROOT / "third_party"
_WEIGHTS_DIR  = _PROJECT_ROOT / "checkpoints" / "medvae"

# Vendor MedVAE library source from third_party/ (no pip install needed)
if (_VENDOR_DIR / "medvae").is_dir() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from medvae.utils.factory import build_model  # noqa: E402


# Latent statistics measured on v2 train (3000 slices), model_name="medvae_4_3_2d"
# See: logs/medvae_latent_stats_*.log
_LATENT_MEAN = torch.tensor([10.8735, -9.2636, -0.8051]).view(1, 3, 1, 1)
_LATENT_STD  = torch.tensor([ 6.5829,  4.9971,  1.4263]).view(1, 3, 1, 1)


# Local-file mapping for each supported variant. Add more entries if you copy
# their .yaml/.ckpt under checkpoints/medvae/.
_LOCAL_FILES = {
    "medvae_4_3_2d": ("medvae_4x3.yaml", "vae_4x_3c_2D.ckpt"),
}


class MedVAEAdapter(nn.Module):
    """Drop-in replacement for `models.vae.VAE` backed by MedVAE 4_3_2d."""

    DEFAULT_MODEL_NAME = "medvae_4_3_2d"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, weights_dir: Path = None):
        super().__init__()
        if model_name not in _LOCAL_FILES:
            raise ValueError(
                f"Unknown MedVAE variant {model_name!r}. Available: {list(_LOCAL_FILES)}"
            )
        weights_dir = Path(weights_dir) if weights_dir is not None else _WEIGHTS_DIR
        cfg_name, ckpt_name = _LOCAL_FILES[model_name]
        config_path = weights_dir / cfg_name
        ckpt_path   = weights_dir / ckpt_name

        for p in (config_path, ckpt_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"MedVAE asset missing: {p}\n"
                    f"Copy from HF cache (one-off):\n"
                    f"  mkdir -p {weights_dir}\n"
                    f"  cp /root/.cache/huggingface/hub/models--stanfordmimi--MedVAE/"
                    f"snapshots/*/model_weights/{p.name} {weights_dir}/"
                )

        self.model_name = model_name
        self.model = build_model(model_name, str(config_path), str(ckpt_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer("latent_mean", _LATENT_MEAN.clone(), persistent=False)
        self.register_buffer("latent_std",  _LATENT_STD.clone(),  persistent=False)

    # ------------------------------------------------------------------
    # Core API (mirrors models.vae.VAE)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """(B, 1, H, W) in [-1, 1] → (mu, logvar) at (B, 3, H/4, W/4), z-scored."""
        if x.size(1) == 1:
            x3 = x.repeat(1, 3, 1, 1)
        elif x.size(1) == 3:
            x3 = x
        else:
            raise ValueError(f"MedVAEAdapter expects 1 or 3 input channels, got {x.size(1)}")

        posterior = self.model.encode(x3)
        mu_norm = (posterior.mean - self.latent_mean) / self.latent_std
        # After (x - μ) / σ, Var(scaled) = Var(raw) / σ² ⇒ logvar shifts by -2·log(σ)
        logvar_norm = posterior.logvar - 2.0 * self.latent_std.log()
        return mu_norm, logvar_norm

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z_norm: torch.Tensor):
        """(B, 3, h, w) z-scored → (B, 1, h*4, w*4) image in [-1, 1]."""
        z_raw = z_norm * self.latent_std + self.latent_mean
        out3 = self.model.decode(z_raw)            # (B, 3, h*4, w*4)
        return out3.mean(dim=1, keepdim=True).clamp(-1.0, 1.0)

    def forward(self, x: torch.Tensor):
        """Match `VAE.forward` signature: returns (z, mu, logvar, recon)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z)
        return z, mu, logvar, recon


def load_medvae(model_name: str = MedVAEAdapter.DEFAULT_MODEL_NAME,
                weights_dir: Path = None,
                device=None) -> MedVAEAdapter:
    """Convenience constructor mirroring `models.vae.load_vae` semantics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MedVAEAdapter(model_name=model_name, weights_dir=weights_dir).to(device).eval()
    return adapter
