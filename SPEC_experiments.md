# 主实验矩阵 SPEC

> **当前状态（2026-05-07）**：Screening 已完成；D1-strong-bc256 50 epoch pilot 完成（MAE 101 HU）；L0-CFG-current 正在运行。
>
> **核心策略**：D1（全模块）优先训练到收敛，以 L0-CFG（无 ControlNet、无 region emb）作充分训练后的消融对照。Screening 结果仅用于快速排除明显无效配置，不作为模块贡献的定量证据——50 epoch latent loss 不能代表 decoded 图像质量，不同架构收敛速度不同。

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

意义：确立 decoded 指标理论下界——扩散模型 decoded MAE 不可能低于此值。若两者接近，瓶颈在 VAE 容量，不是扩散模型。

执行时机：D1-stable-continuation 启动之前（不阻塞训练）。

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

### L0-CFG：消融基线（对照 D1）

```
UNetConcatenation（legacy，base_ch=256）
CFG dropout=0.15 | EMA 0.9999 | bf16 | lr 1e-4 | WSD | bs 42 | epochs 55
pure noise t_start=999 | DDIM 40 步
```

**当前状态**：running（Step 8）。脚本：`scripts/train_legacy_cfg.py`。

```bash
python scripts/train_legacy_cfg.py \
  --manifest data/manifest.csv \
  --vae-path checkpoints/vae/vae_best.pth \
  --save-dir checkpoints/phase1_matrix/L0-CFG-bc256-bs42-ep55-s42 \
  --base-channels 256 --batch-size 42 --epochs 55 --early-stopping 40 \
  --lr 1e-4 --lr-schedule wsd \
  --precision bf16 --ema-decay 0.9999 \
  --cfg-dropout-rate 0.15 \
  --latent-mode mu --loss-scope mask \
  --sampler-init noise --sampler-t-start 999 --ddim-steps 40
```

epochs=55 对应约 30,690 optimizer steps，与旧成功 run（`concatenation_CFG_20260210_103659_ep400`，29,600 step）等价，而非照抄 400 epoch。显存：bs48/75ep 在 epoch 2 OOM（peak 22.1GB），bs42 稳定（peak 19.95GB）。

**对比意义**：D1 有 ControlNet + region embedding；L0-CFG 无。充分训练后的 decoded MAE 差值即两者联合贡献。

注意非单变量差异：L0-CFG 额外有 CFG dropout、不同 sampler（t999 vs t300）。最终比较时两者都需做 sampler sweep，用各自最优 sampler 结果相互对照。

---

### 后续消融（可选）

若需精确拆分 ControlNet 与 region embedding 的独立贡献，追加：

```
D1-no-region = D1 配置，去掉 region embedding，其他完全一致
```

从 D1-strong best EMA 重新训练（不能 fine-tune，region emb 影响 time embedding 维度）。优先级低于 D1/L0-CFG 完成。

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
[running] L0-CFG-current  (bc256/bs42/ep55/WSD)
[next]    VAE 重建基线  — val 23 例 GT CT encode→decode，记录 MAE/PSNR/SSIM 下界
[next]    D1-stable-continuation  — 从 pilot best EMA，lr=3e-5，bf16，ep100
[after]   sampler sweep（D1 + L0-CFG 各自跑）
            t_start ∈ {200, 300, 500, 700, 999} × alpha ∈ {0.7, 1.0}
            t999 = 纯噪声起点，D1 CBCT 条件来自 latent concat 而非 sampler 起点，
            纯噪声推理可能同样有效；L0-CFG 已默认 t999
            用各自最优 sampler 结果做最终 D1 vs L0-CFG 比较
[if needed] D1-no-region  — 拆分 ControlNet vs region embedding 独立贡献
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
3. **D1 vs L0-CFG 非单变量对照**：差异包含 ControlNet、region embedding、CFG dropout 三个变量，以及 sampler 协议（t300 vs t999）。不是精确消融，但是当前可行的最近似对照。sampler sweep 后统一用各自最优 sampler 结果比较。
4. **单 seed**：所有实验基于 seed=42 单次。论文/最终决策前需补 seed=43 复现至少 A0 + D1。
5. **held-out test**：盲提交集（官方 val 42 例）由 D1 完成后单独写 inference 脚本。当前所有指标来自 val=23 例，最终以官方提交为准。
