# 数据预处理 SPEC

> 本文件只保留预处理入口说明。完整数据、VAE、Phase 1 训练和实验记录见 `SPEC_pipeline.md`。

## 当前预处理结论

- 主实现：`scripts/preprocess_synthrad_dataset.py`
- 处理框架：SimpleITK 加载 + MONAI `Compose` / `MapTransform`
- 输入数据：synthRAD2023 Task2 Brain + synthRAD2025 Task2 AB / HN / TH
- 输出目录：`data/preprocessed/`
- manifest：`data/manifest.csv`
- 输出格式：MHA volume，而不是逐 slice NPY

## 核心流程

1. 加载 CT / CBCT / mask，并统一为 `(1, Z, H, W)`。
2. CT / CBCT HU clip 到 `[-1024, 1500]`。
3. 基于 mask union bbox 做前景 crop，margin 为 10 px。
4. XY 等比缩放，长边到 256，短边中心 padding 到 256。
5. 图像归一化到 `[-1, 1]`，mask 二值化。
6. 保存 `ct_preprocessed.mha`、`cbct_preprocessed.mha`、`mask_preprocessed.mha`、`cbct_global.mha` 和 `preprocess_metadata.json`。

## 重要约定

- 当前不做 Z 方向重采样，2D 模型逐 slice 训练。
- 当前不把 mask 外全部强制设为空气；只做 mask union bbox crop + padding air。
- `cbct_global.mha` 仅作为 Phase 2 可选增强备用，不参与当前 Phase 1 baseline。
- `preprocess_metadata.json` 用于把 256x256 输出反向映射回原始网格。

## 完整文档

完整方案、训练状态、WandB 记录、VAE 验证和 Phase 1 训练建议见：

```text
SPEC_pipeline.md
```
