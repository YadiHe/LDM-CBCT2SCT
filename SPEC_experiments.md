# 主实验矩阵 SPEC

> 目标：用可解释的实验顺序验证 Concat latent、DR、ControlNet、PACA 对 CBCT-to-sCT Phase 1 的边际贡献，最终确定一个可靠的 D1 主模型。

---

## 一、共同设置

所有实验默认共享以下设置：

- 数据：synthRAD2023 BB + synthRAD2025 AB / HN / TH
- Manifest：`data/manifest.csv`
- VAE：`checkpoints/vae/vae_best.pth`，冻结
- Latent：默认使用 VAE `mu`，不使用 `reparameterize`
- 输入条件：CBCT latent concat
- Region embedding：开启
- Loss：mask-weighted diffusion MSE
- WandB project：`cbct2sct_IBA`
- 推荐起步显存配置：`base_channels=64, batch_size=4, grad_accum_steps=2`

所有实验必须记录同一批固定 validation case 的图像结果，不能只比较 latent noise loss。

---

## 二、实验矩阵

### A0：主 Baseline

```text
Concat latent + region embedding + mask-weighted diffusion loss + VAE mu
```

目的：

- 建立最小可靠 baseline。
- 验证仅靠 CBCT latent concat 是否能产生可用 sCT。
- 后续所有增强模块都必须和 A0 对比。

### B：DR 模块验证

```text
B1 = A0 + DR, gamma=0.5
B2 = A0 + DR, gamma=1.0
```

目的：

- 判断 pixel-space CBCT 降质建模是否帮助 latent diffusion。
- 比较 `gamma=0.5` 与 `gamma=1.0` 是否存在 DR loss 压过 diffusion loss 的问题。

判断：

- 若 B1/B2 明显优于 A0，则 DR 保留。
- 若 `gamma=1.0` 图像更平滑、diffusion loss 改善更慢或伪影更多，则优先 `gamma=0.5`。
- 若 B1/B2 差异小，优先更稳定的 `gamma=0.5`。

### C：ControlNet / PACA 融合验证

```text
C1 = A0 + ControlNet residual add
C2 = A0 + ControlNet PACA only
C3 = A0 + ControlNet residual add + PACA
```

目的：

- C1 验证标准 ControlNet 多尺度 residual 注入。
- C2 验证 PACA attention 融合是否比直接 residual add 更有效。
- C3 验证当前双注入方式是否有收益。

判断：

- C1 优于 C2：采用标准 ControlNet residual add。
- C2 优于 C1：采用 PACA only，避免双注入。
- C3 明显优于 C1/C2：保留双注入，但需要记录显存和推理成本。

### D1：最终主模型

```text
D1 = A0 + best DR gamma + best ControlNet/PACA fusion
```

目的：

- 用前面筛选出的最佳模块组合进行完整长训。
- D1 才作为主实验最终模型。

---

## 三、训练轮数策略

不要一开始每个组合都跑满 300 epoch。建议采用三层训练长度：

### 3.1 Smoke Test：1-5 epoch

用途：

- 检查代码路径、显存、WandB、loss 是否正常。
- 不用于判断模型优劣。

每个新结构至少先跑：

```text
1-5 epoch
max_train_batches 可限制到 20-100
max_val_batches 可限制到 5-20
```

通过标准：

- 无 OOM / NaN。
- train loss 和 val loss 能正常记录。
- fixed val 图像能上传。

### 3.2 Screening Run：30-50 epoch

用途：

- 用较低成本筛选 A/B/C 阶段组合。
- 判断是否值得进入长训。

建议：

```text
A0：50 epoch
B1/B2：各 50 epoch
C1/C2：各 50 epoch
C3：30-50 epoch
```

选择 50 epoch 的原因：

- 5 epoch 只能证明链路可跑，不能判断模块贡献。
- 30 epoch 通常能看到 loss 趋势和图像质量初步差异。
- 50 epoch 更适合比较 val loss、MAE/SSIM 和固定样本可视化。
- 不建议筛选阶段直接跑 200-300 epoch，成本太高。

筛选时可以设置：

```text
early_stopping = 15-20
```

若 20-30 epoch 内明显发散、图像明显劣化或 val loss 长期不如 A0，可提前停止。

### 3.3 Long Run：200-300 epoch

用途：

- 只给最终 D1 主模型。
- 必要时给 A0 也跑一个长训，作为最终 baseline 对照。

建议：

```text
D1：200-300 epoch
A0-long：可选，200 epoch
early_stopping = 40-50
```

如果 D1 在 150-200 epoch 后仍持续改善，可以继续到 300 epoch。若 40-50 epoch 无 val 改善，则 early stop。

---

## 四、推荐训练顺序

1. 先跑 A0，确认主 baseline。
2. 跑 B1/B2，确定 DR 是否值得保留和 gamma。
3. 跑 C1/C2，确定 ControlNet 融合方式。
4. C3 作为当前全组件版对照。
5. 用最佳组合跑 D1 长训。

推荐最低成本顺序：

```text
A0  50 epoch
B1  50 epoch
B2  50 epoch
C1  50 epoch
C2  50 epoch
C3  30-50 epoch
D1  200-300 epoch
```

如果资源紧张：

```text
A0  50 epoch
B1  30 epoch
B2  30 epoch
C1  30 epoch
C2  30 epoch
D1  200 epoch
```

---

## 五、验收指标

每个实验至少记录：

```text
train_loss
val_loss
train/loss_diff
val/loss_diff
train/loss_dr
val/loss_dr
fixed val CT / CBCT / sCT / error map
MAE / PSNR / SSIM
GPU memory
epoch time
```

说明：

- 最终不只看 latent loss，必须看 decoded sCT 图像质量。
- DR 相关指标只对带 DR 的实验有意义；A0/C1/C2 如果不启用 DR，可记录为 0 或 N/A。
- `val/loss_diff` 只说明 noise prediction，不等价于最终 sCT 质量。
- fixed val batch 必须跨实验保持一致，便于直接比较。

---

## 六、模型选择规则

优先级从高到低：

1. 固定 val 图像是否减少明显伪影、床板/固定装置误生成和边界错误。
2. MAE / PSNR / SSIM 是否优于 A0。
3. `val_loss` 和 `val/loss_diff` 是否稳定下降。
4. 训练是否稳定，无 NaN/OOM。
5. 显存和 epoch time 是否可接受。

如果指标冲突：

- 图像质量和 MAE 优先于 latent loss。
- 稳定模型优先于偶然最低 val loss。
- 简单模型优先于复杂模型，除非复杂模型有明确收益。

---

## 七、待实现能力

为了按本 SPEC 执行，还需要补齐：

- `train_concat_paca.py` 支持选择模块组合：A0 / DR / ControlNet add / PACA only / add+PACA。
- Phase 1 固定 val 可视化：CBCT、target CT、predicted sCT、absolute error map。
- Phase 1 decoded sCT 指标：MAE、PSNR、SSIM。
- 记录 GPU memory 和 epoch time 到 WandB。
- 训练 diffusion 时支持 `--latent-mode mu|sample`，默认 `mu`。
