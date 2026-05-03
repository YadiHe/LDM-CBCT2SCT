# LDM-CBCT2SCT

基于潜在扩散模型将 CBCT 转换为 synthetic CT，用于 synthRAD CBCT-to-sCT 任务。

当前主线已经迁移到：

- MONAI + SimpleITK 预处理
- MHA volume + `data/manifest.csv`
- CT-only VAE
- 冻结 VAE 后训练 ConcatPACA Phase 1
- WandB project：`cbct2sct_IBA`

完整设计和实验记录见：

- `SPEC_preprocessing.md`：预处理 SPEC
- `SPEC_pipeline.md`：数据、VAE、Phase 1 训练和调试记录
- `scripts/README.md`：当前脚本入口说明

## 当前主入口

```text
scripts/preprocess_synthrad_dataset.py  # synthRAD2023/2025 预处理
scripts/train_ct_vae.py                 # CT-only VAE 训练和 eval-only 验证
scripts/train_concat_paca.py            # ConcatPACA Phase 1 训练
```

旧实验、旧推理、旧评估和论文制图脚本已归档到：

```text
scripts/legacy/
```

这些 legacy 脚本未随当前 MONAI + MHA pipeline 重新验证。

## 数据预处理

```bash
python scripts/preprocess_synthrad_dataset.py \
  --raw-dir rawdata \
  --out-dir data/preprocessed \
  --manifest data/manifest.csv
```

输出：

```text
data/preprocessed/{patient_id}/
  ct_preprocessed.mha
  cbct_preprocessed.mha
  mask_preprocessed.mha
  cbct_global.mha
  preprocess_metadata.json

data/manifest.csv
```

## VAE 训练

```bash
python scripts/train_ct_vae.py \
  --manifest data/manifest.csv \
  --save-dir checkpoints/vae \
  --batch-size 16 \
  --num-workers 4 \
  --wandb-project cbct2sct_IBA
```

验证已训练好的 VAE：

```bash
python scripts/train_ct_vae.py \
  --manifest data/manifest.csv \
  --save-dir checkpoints/vae_eval_best \
  --resume checkpoints/vae/vae_best.pth \
  --eval-only \
  --no-amp \
  --wandb-project cbct2sct_IBA \
  --wandb-name vae-best-val-eval
```

## Phase 1 训练

```bash
python scripts/train_concat_paca.py \
  --manifest data/manifest.csv \
  --vae-path checkpoints/vae/vae_best.pth \
  --save-dir checkpoints/concat_paca \
  --base-channels 64 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --lr 5e-6 \
  --wandb-project cbct2sct_IBA
```

当前建议先用 `base_channels=64` 建立稳定 baseline；`base_channels=128/256` 在单卡 4090 上需要另行测显存和稳定性。
