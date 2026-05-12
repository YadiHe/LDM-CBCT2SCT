#!/usr/bin/env bash
# 2.5D U-Net++ ResNet34 direct CT regression baseline.

set -euo pipefail

RUN_NAME="UNetPP25D-resnet34-s5-bs128-l1grad-full005-msssim010-hd020hu500-s42"
SAVE_DIR="checkpoints/unetpp_25d/${RUN_NAME}"
WARMSTART_CKPT="checkpoints/unetpp_25d/UNetPP25D-resnet34-s5-bs128-l1grad-s42/model_latest.pth"
RESUME_ARG=""
RESUME_CKPT=""

if [[ -f "${SAVE_DIR}/model_latest.pth" ]]; then
  RESUME_CKPT="${SAVE_DIR}/model_latest.pth"
elif [[ -f "${WARMSTART_CKPT}" ]]; then
  RESUME_CKPT="${WARMSTART_CKPT}"
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_ARG="--resume ${RESUME_CKPT}"
fi

mkdir -p logs "${SAVE_DIR}"

screen -dmS unetpp25d bash -lc "
  python -u unetpp_25d/train.py \
    --manifest data/manifest.csv \
    --save-dir ${SAVE_DIR} \
    --input-slices 5 \
    --batch-size 128 \
    --num-workers 4 \
    --encoder-name resnet34 \
    --encoder-weights imagenet \
    --epochs 400 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --grad-weight 0.05 \
    --full-mae-weight 0.05 \
    --ms-ssim-weight 0.10 \
    --high-density-weight 0.20 \
    --high-density-threshold-hu 500 \
    --amp-dtype fp16 \
    --eval-every 5 \
    --wandb-project cbct2sct_25unetPP_IBA \
    --wandb-group unetpp25d-v2 \
    --wandb-name ${RUN_NAME} \
    --seed 42 \
    ${RESUME_ARG} \
    2>&1 | tee logs/${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log
"

echo "UNet++ 2.5D launched in screen session: unetpp25d"
echo "Monitor: screen -r unetpp25d"
echo "Log:     ls logs/${RUN_NAME}_*.log"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "Resume:  ${RESUME_CKPT}"
fi
