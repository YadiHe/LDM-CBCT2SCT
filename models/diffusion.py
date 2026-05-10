import math
import torch


def _cosine_beta_schedule(timesteps, s=0.008):
    """iDDPM cosine schedule (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)


class Diffusion:
    def __init__(self, device, timesteps=1000, beta_start=0.0001, beta_end=0.02,
                 noise_schedule="linear"):
        self.timesteps = timesteps
        self.device = device

        if noise_schedule == "cosine":
            betas = _cosine_beta_schedule(timesteps).to(device)
        elif noise_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        else:
            raise ValueError(f"noise_schedule must be 'linear' or 'cosine', got {noise_schedule!r}")

        self.beta = betas
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def sample_timesteps(self, batch_size, generator=None):
        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            generator=generator,
            device=self.device,
        )

    def sample_timesteps_logit_normal(self, batch_size, mean=0.0, std=1.0, generator=None):
        """Logit-normal timestep sampling (SD3 style).
        Concentrates training on middle timesteps where perceptual quality matters most.
        u ~ N(mean, std²), t = sigmoid(u) * T, clipped to [0, T-1].
        """
        u = torch.randn(batch_size, device=self.device, generator=generator) * std + mean
        t = (torch.sigmoid(u) * self.timesteps).long().clamp(0, self.timesteps - 1)
        return t
