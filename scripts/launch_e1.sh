#!/usr/bin/env bash
# E1: 全模块全开 + cosine noise schedule + SOTA LR (1e-4 cosine warmup)
# 模块: UNet(bc256) + DR + ControlNet(cbct_source=dr, fusion=add) + EMA
# 噪声: cosine beta (iDDPM, Nichol & Dhariwal 2021), T=1000
# 优化: AdamW lr=1e-4, sd-warmup-cosine, warmup=1000, min_lr=0.1x
# 损失: L1 + Min-SNR-γ=5 (Hang et al. 2023)
# AMP: bf16
# 对照: D1-rollback (MSE, linear, lr=3e-5, no DR)
set -euo pipefail

SAVE_DIR="checkpoints/phase1_matrix/E1-cosine-fullpaca-bc256-bs24-ep100-s42"

screen -dmS e1_cosine_full bash -lc "
  HTTP_PROXY=http://127.0.0.1:7892 HTTPS_PROXY=http://127.0.0.1:7892 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -u scripts/train_concat_paca.py \
    --manifest data/manifest.csv \
    --vae-path checkpoints/vae/vae_best.pth \
    --save-dir ${SAVE_DIR} \
    --use-dr \
    --use-controlnet --control-source dr --controlnet-fusion both \
    --base-channels 256 \
    --use-ema --ema-decay 0.9995 \
    --dropout-rate 0.1 \
    --lr 1e-4 \
    --lr-schedule sd-warmup-cosine \
    --warmup-steps 1000 \
    --cosine-min-lr-ratio 0.1 \
    --noise-schedule cosine \
    --prediction-type v_prediction \
    --timestep-sampling logit_normal \
    --latent-scale 1.3995 \
    --loss-type l1 \
    --use-min-snr-weight --min-snr-gamma 5.0 \
    --amp-dtype bf16 \
    --batch-size 24 \
    --epochs 100 \
    --gamma 0.5 \
    --sampler-init noise --sampler-t-start 999 \
    --ddim-steps 100 --eval-every 10 \
    --exp-id E1-cosine-fullpaca --stage long \
    --wandb-group phase1-matrix-2026-05 \
    --wandb-name E1-cosine-fullpaca-bc256-bs24-ep100-s42 \
    --seed 42 \
    2>&1 | tee logs/e1_cosine_full_\$(date +%Y%m%d_%H%M%S).log
"

echo "E1 launched in screen session: e1_cosine_full"
echo "Monitor: screen -r e1_cosine_full"
echo "Log: ls logs/e1_cosine_full_*.log"
