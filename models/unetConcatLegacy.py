"""Legacy concat UNet for CBCT-to-sCT.

Faithful port of the legacy ``UNetConcatenation`` (see
``models/unetConditional.py``) with the following deliberate differences vs
``UNetConcatControlPACA``:

- no ControlNet residual hooks (down/middle injection points removed)
- no PACA layers in up blocks (uses plain ``UpBlock``)
- no region embedding
- dropout default 0.1 (matches legacy training)

The forward signature accepts (and ignores) the ControlNet/PACA/region kwargs
so this module is drop-in compatible with ``train_unet_concat_control_paca``
when called via ``--use-controlnet=False --no-use-dr``.
"""

import torch
import torch.nn as nn

from models.blocks import (
    DownBlock,
    MiddleBlock,
    Normalize,
    TimestepEmbedding,
    UpBlock,
    nonlinearity,
)


class UNetConcatLegacy(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 base_channels=256,
                 dropout_rate=0.1):
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
        self.init_conv = nn.Conv2d(in_channels * 2, ch1, kernel_size=3, padding=1)

        self.down1 = DownBlock(ch1, ch1, time_emb_dim, attn_res_64, dropout_rate)
        self.down2 = DownBlock(ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.down3 = DownBlock(ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.down4 = DownBlock(ch3, ch4, time_emb_dim, attn_res_8, dropout_rate, downsample=False)

        self.middle = MiddleBlock(ch4, time_emb_dim, dropout_rate)

        self.up4 = UpBlock(ch4, ch3, ch4, time_emb_dim, attn_res_8, dropout_rate)
        self.up3 = UpBlock(ch3, ch2, ch3, time_emb_dim, attn_res_16, dropout_rate)
        self.up2 = UpBlock(ch2, ch1, ch2, time_emb_dim, attn_res_32, dropout_rate)
        self.up1 = UpBlock(ch1, ch1, ch1, time_emb_dim, attn_res_64, dropout_rate, upsample=False)

        self.final_norm = Normalize(ch1)
        self.final_conv = nn.Conv2d(ch1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, condition, t, **_ignored):
        # _ignored swallows down_paca_control_residuals / middle_paca_control_residual /
        # down_control_residuals / middle_control_residual / region_id / controlnet_fusion
        # so the trainer can call this module with the same signature as UNetConcatControlPACA.
        t_emb = self.time_embedding(t)
        x = torch.cat((x, condition), dim=1)

        h = self.init_conv(x)
        h, intermediates1 = self.down1(h, t_emb)
        h, intermediates2 = self.down2(h, t_emb)
        h, intermediates3 = self.down3(h, t_emb)
        h, intermediates4 = self.down4(h, t_emb)

        h = self.middle(h, t_emb)

        h = self.up4(h, intermediates4, t_emb)
        h = self.up3(h, intermediates3, t_emb)
        h = self.up2(h, intermediates2, t_emb)
        h = self.up1(h, intermediates1, t_emb)

        h = self.final_norm(h)
        h = nonlinearity(h)
        h = self.final_conv(h)
        return h


def load_unet_concat_legacy(unet_save_path=None, base_channels=256, dropout_rate=0.1, trainable=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetConcatLegacy(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
    ).to(device)

    if unet_save_path is None:
        print("UNetConcatLegacy initialized with random weights.")
    else:
        state = torch.load(unet_save_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  UNetConcatLegacy missing keys  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  UNetConcatLegacy unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"UNetConcatLegacy loaded from {unet_save_path}")

    for p in model.parameters():
        p.requires_grad = trainable
    return model
