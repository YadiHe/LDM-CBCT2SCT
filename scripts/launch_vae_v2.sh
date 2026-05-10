#!/usr/bin/env bash
# VAE-v2：在 prep-v2 数据上重训 CT VAE
#
# 重训动机（SPEC §6.1 P2）：
#   旧 VAE（v1）在 v2 数据上 mae_hu_vae = 58 HU（v1 数据上是 25 HU），
#   超过了 10 HU 的退化阈值。原因：
#     1. CLIP_MAX 1500→2000，HU 轴拉伸约 1.2 倍
#     2. mask 外被强制置为 -1.0，编码器输入分布漂移
#     3. 1500–2000 HU 段（致密骨/牙）旧 VAE 训练时被 clip 掉，从未见过
#
# 架构：保持 v1 不变（base_ch=64, latent=3），D1/E1 latent shape 不变，
#       新 VAE 训完后可直接替换给下游用。
# 保存：checkpoints/vae_v2/ —— 不动旧的 checkpoints/vae/vae_best.pth (v1)。
# 评估：每 10 epoch 跑一次 patient-level 指标（utils.image_metrics.ImageMetrics）：
#         mae_hu_vae    : mask 内 HU-空间 MAE（mask=0 voxel 不计入分母）
#         psnr_vae      : mask 内 PSNR，data_range = 4024 HU（SynthRAD 官方动态范围 [-1024, 3000]）
#         ms_ssim_vae   : 5-scale 3D MS-SSIM，masked 版本（每 scale 的 SSIM map 在 mask 内取均值再连乘）
#       三个指标均与 D1 训练 val 同标尺、与 SynthRAD 官方排行榜对齐。
# 参考：v1 VAE 在 v1 数据上 ≈ 25 HU；v2 训练目标：受 CLIP 缩放影响后 ≤ 30 HU。

set -euo pipefail

SAVE_DIR="checkpoints/vae_v2"
RUN_NAME="vae-v2-bc64-ep200"

mkdir -p logs "${SAVE_DIR}"

screen -dmS vae_v2 bash -lc "
  HTTP_PROXY=http://127.0.0.1:7892 HTTPS_PROXY=http://127.0.0.1:7892 \
  python -u scripts/train_ct_vae.py \
    --manifest data/manifest.csv \
    --save-dir ${SAVE_DIR} \
    --base-channels 64 \
    --latent-channels 3 \
    --batch-size 16 \
    --num-workers 4 \
    --epochs 200 \
    --lr 6.25e-6 \
    --early-stopping 30 \
    --patience 10 \
    --l1-weight 1.0 \
    --mse-weight 0.0 \
    --ssim-weight 0.8 \
    --perceptual-weight 0.1 \
    --kl-weight 1e-5 \
    --vis-every 10 \
    --vis-num-samples 4 \
    --wandb-project cbct2sct_IBA \
    --wandb-name ${RUN_NAME} \
    2>&1 | tee logs/${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log
"

echo "VAE-v2 已在 screen 会话启动：vae_v2"
echo "查看进度： screen -r vae_v2"
echo "日志路径： ls logs/${RUN_NAME}_*.log"
echo "WandB：    https://wandb.ai/SMU-BME/cbct2sct_IBA  (run name: ${RUN_NAME})"
echo
echo "决策点（SPEC §6.1，均为 mask 内指标）："
echo "  - ep10 mae_hu_vae < 35 HU  → 训练正常，继续"
echo "  - ep30 mae_hu_vae < 28 HU  → 路径 A：长训到 ep200"
echo "  - ep30 mae_hu_vae > 40 HU  → 架构 review（kl_weight / latent_channels）"
