# 主实验矩阵 SPEC

> **当前状态（2026-05-07）**：Screening 已完成；D1-strong-bc256 50 epoch pilot 完成（MAE 101 HU）；L0-CFG-current 已完成，decoded 质量很差，不作为消融基线。
>
> **立即下一步**：**先跑 VAE 重建基线**（§二），确认 VAE 不是当前 MAE 101 HU 的瓶颈，再启动 D1-stable-continuation。
>
> **核心策略**：D1（全模块全容量）为基准，逐模块关闭做 top-down 消融（D1-no-controlnet、D1-no-region）。消融实验与 D1 使用完全相同的训练协议，保证公平对比。Screening 结果仅用于快速排除明显无效配置，不作为模块贡献的定量证据。

---

## 一、共同设置

### 数据与 VAE

- 数据：synthRAD2023 BB + synthRAD2025 AB / HN / TH
- Manifest：`data/manifest.csv`（train 218 / val 23，patient-level split）
- VAE：`checkpoints/vae/vae_best.pth`，**冻结**，Phase 1 内不重训
- Latent 模式：VAE `mu`（不 reparameterize）
- Loss：mask-weighted diffusion MSE（latent 空间，mask 由 256→64 avg-pool）

### D1-strong 训练配置

| 项 | 值 |
|---|---|
| 架构 | UNet + ControlNet（CBCT latent ZeroConv adapter, residual add）+ region embedding |
| `base_channels` | 256 |
| EMA | on，`ema_decay=0.9995` |
| AMP | bf16 优先；fp16 下监控 `skipped_overflow_steps` |
| Optimizer | AdamW，`wd=1e-4` |
| LR（continuation） | `3e-5`，warmup 1000 step → constant |
| `batch_size` | 24（bf16 下约 19 GB peak）；OOM 则降 20 |
| Seed | 42 |
| `num_workers` | 4 |

### WandB 与命名

- Project：`cbct2sct_IBA`，Group：`phase1-matrix-2026-05`
- Run name：`{exp_id}-bc{base_channels}-ep{epochs}-s{seed}`
- Checkpoints：`checkpoints/phase1_matrix/{run_name}/`
- Tags：`exp_id`、`stage`（smoke/screen/strong）、`commit`（git SHA）

---

## 二、评估协议

**核心原则**：`val/loss_diff`（latent 扩散 loss）**不作为模型排序依据**，仅监控训练稳定性。主排序指标是 decoded sCT 在 HU 空间的 MAE。

### 前置：VAE 重建基线

在报告任何扩散模型 decoded 指标之前，先对 val 23 例 GT CT 做 `VAE encode（mu）→ decode`，记录：

- `mae_hu_vae / psnr_vae / ssim_vae`（整体 + 分 region AB/BB/HN/TH）

意义：确立 decoded 指标理论下界——扩散模型 decoded MAE 不可能低于此值。

**结果（2026-05-07，val 23 例，2487 slices）**：

| 指标 | 整体 | AB | BB | HN | TH |
|---|---:|---:|---:|---:|---:|
| `mae_hu_vae` | **24.99 HU** | 20.98 | 29.11 | 22.10 | 22.80 |
| `psnr_vae` | 35.19 dB | — | — | — | — |
| `ssim_vae` | 0.9778 | — | — | — | — |

**结论**：VAE 不是瓶颈。VAE 重建 MAE 25 HU，D1 pilot 为 101 HU，gap = 76 HU，扩散模型有大量提升空间。继续训练 D1。D1 的理论上限约 25 HU（VAE 重建下界）。

### Fixed val 定性检查

- `configs/fixed_val_cases.yaml`：每个 region 4 例（共 16 例），固定 z 索引（mask 前景面积最大的 3 个 slice）
- 推理：固定 seed=0；CBCT-init `t_start=300`，DDIM 50 步
- 用途：跨 run 可视化对比，定性排查伪影；不参与定量排序

### 完整 val 集指标

每个 epoch 末记录：

| 指标 | 范围 | 说明 |
|---|---|---|
| `val/mae_hu` | mask 内，HU | **主排序**，反归一化回 HU |
| `val/psnr` | mask 内，`data_range=2` | [-1,1] 空间 |
| `val/ssim` | mask 内，`data_range=2`，window=11 | [-1,1] 空间 |
| `val/mae_hu_{AB,BB,HN,TH}` | 同上 | 分 region 监控 |
| `val/loss_diff` | latent | 稳定性参考，不排序 |

sCT 重建：`CBCT latent 加噪到 t_start=300 → DDIM 50 步 → VAE decode → 反归一化到 HU`

资源指标（每个 run 必须记录）：`gpu_mem_max_gb`、`epoch_time_sec`、`step_time_ms`、活跃参数量

---

## 三、实验

### D1-strong：主模型（当前优先目标）

```
UNet（base_ch=256）+ ControlNet（CBCT latent adapter, residual add）+ region embedding
EMA 0.9995 | bf16 | lr 3e-5 | WSD warmup→constant | bs 24
```

**Pilot 结果（总 epoch ~50，W&B: `ab94l2m1`）**：

| 指标 | 数值 |
|---|---:|
| `val/mae_hu` | 101.01 |
| `val/psnr` | 23.39 |
| `val/ssim` | 0.8796 |
| MAE AB / BB / HN / TH | 128.70 / 88.84 / 114.22 / 84.52 |

训练历史：原始 bs24 run（`iodr4amf`）在 epoch 22 出现 NaN，从 epoch 22 EMA 恢复后 bs20 + lr=5e-5 续训 28 epoch；期间 `grad_norm=inf` 跳步但无 NaN。Best checkpoint 在恢复 epoch 18（总 epoch ~40）。

**D1-stable-continuation 启动命令**：

```bash
python scripts/train_concat_paca.py \
  --use-controlnet --control-source cbct_latent --controlnet-fusion add \
  --base-channels 256 \
  --use-ema --ema-decay 0.9995 \
  --lr 3e-5 --lr-schedule sd-warmup-constant --warmup-steps 1000 \
  --amp-dtype bf16 --batch-size 24 \
  --epochs 100 \
  --sampler-init cbct --sampler-t-start 300 --sampler-alpha 1.0 \
  --resume checkpoints/phase1_matrix/D1-strong-bc256-bs20-resume-ep28-s42/unet_ema_state.pth
```

从 pilot best EMA checkpoint（恢复 epoch 18）继续。监控要点：
- 若 `grad_norm=inf` 频繁（>1次/epoch）→ 先降 lr 到 2e-5
- 每 10 epoch 看 decoded MAE/PSNR/SSIM 和 fixed val 图像；若平台 → 做 sampler sweep，不要急着加模块
- 若 decoded 指标在 ep100 仍无改善 → 做 sampler sweep 再判断，不假设模型容量不够

**停止条件**：
- 任意 epoch NaN → 立刻停，降 lr 重启
- ep 100 MAE > 100 HU（pilot 已达 101）→ 排查 lr / EMA / sampler，不继续到 400 epoch
- epoch time × 总 epoch 超出算力预算 → 评估是否降到 D1-128

---

### L0-CFG：已完成，质量差，不作消融基线

`scripts/train_legacy_cfg.py`（bc256/bs42/ep55/WSD/CFG-dropout0.15/t999/DDIM40）已完成训练，decoded 质量很差。

**结论**：旧版 UNetConcatenation + CFG 方案在当前数据集上不可用，与旧数据集上的成功结果不可复现。不作为 D1 的消融对照——两者训练协议差异太多（CFG、sampler、架构），无法干净地归因。

**消融改用 D1 自身去模块对比**（见下）。

---

### D1 模块消融：top-down 去模块对比

以 D1-full 为基准，逐模块关闭，**其他所有训练配置完全一致**（bc256、EMA 0.9995、bf16、lr 3e-5、相同 epoch budget）。

| 实验 | 关闭模块 | 回答的问题 |
|---|---|---|
| **D1-full**（已有 pilot） | — | 基准 |
| **D1-no-controlnet** | ControlNet 关闭（仅保留 CBCT latent concat + region emb） | ControlNet 的边际贡献 |
| **D1-no-region** | region embedding 关闭（保留 ControlNet） | region embedding 的边际贡献 |
| **D1-baseline** | 两者都关（仅 CBCT latent concat） | 联合贡献；与 A0-bc256 等价但训练预算一致 |

**执行优先级**：D1-stable-continuation 完成后，先跑 D1-no-controlnet（最重要的变量），再视时间决定是否补 D1-no-region 和 D1-baseline。每个消融实验使用与 D1-continuation 相同的 checkpoint 起点（从头训练，不 fine-tune）和相同 epoch 数。

启动命令模板（以 D1-no-controlnet 为例）：

```bash
python scripts/train_concat_paca.py \
  --no-use-controlnet \           # 关闭 ControlNet
  --base-channels 256 \
  --use-ema --ema-decay 0.9995 \
  --lr 3e-5 --lr-schedule sd-warmup-constant --warmup-steps 1000 \
  --amp-dtype bf16 --batch-size 24 \
  --epochs 100 \
  --sampler-init cbct --sampler-t-start 300
```

---

## 四、Screening 结果（快速否决参考）

> 以下结果仅用于排除明显无效配置，不作为模块边际贡献的定量证据。

| 实验 | 架构 | epoch | MAE(HU) | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| A0 | concat only | 50 | 156.69 | 20.68 | 0.817 |
| C1 | +ControlNet add | 50 | 144.54 | 21.31 | 0.830 |
| C2 | +ControlNet PACA only | 50 | 149.92 | 21.07 | 0.819 |
| C3 | +ControlNet add+PACA | 50 | 143.90 | 21.37 | 0.827 |
| B1 | +DR+ControlNet add γ=0.5 | 50 | 145.07 | 21.29 | 0.827 |
| B2 | +DR+ControlNet add γ=1.0 | 20 | 188.96 | 19.30 | 0.789 |

分 region MAE（HU）：

| 实验 | AB | BB | HN | TH |
|---|---:|---:|---:|---:|
| A0 | 212.73 | 136.59 | 168.91 | 131.87 |
| C1 | 202.72 | 124.64 | 154.01 | 120.78 |
| C2 | 197.54 | 135.76 | 158.14 | 124.53 |
| C3 | 190.34 | 129.36 | 151.65 | 121.33 |
| B1 | 197.97 | 130.08 | 152.29 | 117.63 |
| B2 | 263.72 | 171.42 | 186.06 | 158.96 |

快速结论：B2 γ=1.0 否决；DR source（B1）相对 C1 无显著优势且引入 aux loss；PACA-only（C2）弱于 add；C1 vs C3 差 0.64 HU（未达 5 HU 阈值），选 C1。因此 D1 用 C1 架构。

---

## 五、执行状态

```text
[done]    A0/B/C/C1/C2/C3 screening
[done]    D1-strong-bc256 50 epoch pilot  (MAE 101, W&B: ab94l2m1)
[done]    L0-CFG-current  (bc256/bs42/ep55) → 质量差，不作消融基线

[done]    VAE 重建基线  mae_hu_vae=25 HU，gap=76 HU，VAE 不是瓶颈，继续 D1

[NEXT]    D1-stable-continuation
              从 pilot best EMA（恢复 epoch 18），lr=3e-5，bf16，ep100
              每 10 epoch 看 decoded MAE 和 fixed val 图像

[then]    sampler sweep（D1）
              t_start ∈ {200, 300, 500, 700, 999} × alpha ∈ {0.7, 1.0}
              t999 = 纯噪声起点，D1 CBCT 条件来自 latent concat 而非 sampler 起点，需实测

[then]    D1 模块消融（按优先级）
              1. D1-no-controlnet  — 最重要，测 ControlNet 边际贡献
              2. D1-no-region      — 测 region embedding 边际贡献
              3. D1-baseline       — 两者都关，与 A0-bc256 等价
```

---

## 六、模型选择规则

优先级从高到低，前面判定即生效：

1. **图像质量**：床板/固定装置误生成、明显伪影、边界错位 → 出现即否决
2. **`val/mae_hu`**（整体 + 4 region）：差异 ≥ 5 HU 视为有意义
3. **`val/ssim`**：差异 ≥ 0.01 视为有意义
4. `val/loss_diff`：仅训练稳定性参考，**不参与排序**
5. 训练稳定性：无周期性 NaN / OOM
6. 资源开销：epoch time ≤ 1.5× A0 可接受

冲突规则：简单模型与复杂模型持平 → 优先简单；某个 region 退化 ≥ 20 HU → 看 fixed val 图像，记录但不立即否决。

---

## 七、产物清单

```text
checkpoints/phase1_matrix/{run_name}/
  ├── unet_full.pth          # raw UNet 权重
  ├── unet_ema.pth           # EMA 权重（--use-ema 时）
  ├── unet_ema_state.pth     # EMA state，供续训
  └── paca_layers.pth        # adapter/PACA 相关权重
```

WandB 为主要审计来源；所有 run 必须有完整 decoded metrics、fixed val 图像、loss 曲线。

---

## 八、已知局限

1. **VAE 冻结**：`vae_best.pth` 基于旧 train=199 split；val 23 例不变，指标横向可比。Phase 1 内不重训。
2. **PACA 死权重**：D1 `fusion=add` 下 PACA 参数（~200M）不参与 forward，仅占存储和 optimizer 状态。活跃参数约 440M（UNet ~290M + ControlNet ~150M + adapter ~0.05M）。**报数时使用活跃参数**。
3. **D1 模块消融非独立**：D1-no-controlnet 和 D1-no-region 各自从头训练，每次只改一个变量，其他配置与 D1-full 完全一致。但两者的 checkpoint 起点不同（D1-full 从 pilot EMA 继续，消融实验从头），如果 D1-full 的 pilot 初始化带来额外好处，消融结果可能偏保守。这是当前最可行的设计。
4. **单 seed**：所有实验基于 seed=42 单次。论文/最终决策前需补 seed=43 复现至少 A0 + D1。
5. **held-out test**：盲提交集（官方 val 42 例）由 D1 完成后单独写 inference 脚本。当前所有指标来自 val=23 例，最终以官方提交为准。
