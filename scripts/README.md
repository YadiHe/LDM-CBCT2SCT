# Scripts

当前推荐主流程只使用 `scripts/` 根目录下的三个入口：

```text
preprocess_synthrad_dataset.py  # synthRAD2023/2025 原始数据预处理，输出 MHA + manifest
train_ct_vae.py                 # CT-only VAE 训练 / eval-only 验证
train_concat_paca.py            # 冻结 VAE 后训练 ConcatPACA Phase 1
```

## 推荐顺序

```bash
python scripts/preprocess_synthrad_dataset.py \
  --raw-dir rawdata \
  --out-dir data/preprocessed \
  --manifest data/manifest.csv

python scripts/train_ct_vae.py \
  --manifest data/manifest.csv \
  --save-dir checkpoints/vae \
  --wandb-project cbct2sct_IBA

python scripts/train_concat_paca.py \
  --manifest data/manifest.csv \
  --vae-path checkpoints/vae/vae_best.pth \
  --save-dir checkpoints/concat_paca \
  --base-channels 64 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --wandb-project cbct2sct_IBA
```

## Legacy

`scripts/legacy/` 内是旧实验、旧推理、旧评估、论文制图和辅助转换脚本。这些脚本未随当前 MONAI + MHA + manifest pipeline 重新验证。

新实验不要直接基于 legacy 脚本扩展，除非先迁移到当前数据接口。

`scripts/visualize/` 内是独立可视化辅助脚本，不作为训练主入口。
