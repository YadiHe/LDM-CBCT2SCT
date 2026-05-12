import torch
import torch.nn as nn


class UNetPlusPlus25D(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        clamp_output: bool = True,
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required. Install with: "
                "pip install segmentation-models-pytorch"
            ) from exc

        weights = None if encoder_weights in ("none", "None", "") else encoder_weights
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
        self.clamp_output = bool(clamp_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return torch.tanh(y) if self.clamp_output else y
