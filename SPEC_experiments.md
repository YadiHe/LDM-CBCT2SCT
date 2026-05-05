# 主实验矩阵 SPEC

> 目标：用可解释的实验顺序验证 Concat latent、DR、ControlNet、PACA 对 CBCT-to-sCT Phase 1 的边际贡献，最终确定一个可靠的 D1 主模型。
>
> 全部实验共享同一份预处理（见 `SPEC_pipeline.md`）和同一组 fixed val cases（见 §二）。每条实验必须可被另一个人按本 SPEC 重跑得到可比较的结果。
>
> 2026-05-05 更新：A/B/C screening 已完成，当前阶段从“模块消融”切换为“D1-strong 主提交候选训练”。`L0-strong`（legacy concat）需要在当前数据集上重训才公平，因此后置为复现实验/论文对照，不阻塞 D1。

---

## 一、共同设置

### 1.1 数据与模型默认

- 数据：synthRAD2023 BB + synthRAD2025 AB / HN / TH
- Manifest：`data/manifest.csv`（patient-level：train 218 / val 23）
- VAE：`checkpoints/vae/vae_best.pth`，**冻结**，不在本批实验中重训
- Latent 模式：使用 VAE `mu`（不 reparameterize），不做消融
- Region embedding：始终开启，不做消融
- 主 loss：mask-weighted diffusion MSE（latent 空间，mask 由 256→64 avg-pool 得到）
- Inference sampler：DDIM 50 步，η=0（确定性），默认 `sampler_init=cbct, sampler_t_start=300, sampler_alpha=1.0`；评估默认用此设置（见 §二）

### 1.2 训练配置（screening 阶段必须一致）

| 项 | 值 | 说明 |
|---|---|---|
| `base_channels` | 64 | A/B/C 全部固定；D1 长训可单独调高 |
| **effective batch** | **64** | 跨实验固定的公平性约束；通过 micro-batch × grad accumulation 达成 |
| `batch_size` × `grad_accum_steps` | 见下表 | 按各实验显存裕度选择，乘积=64 |
| AMP | on（autocast + GradScaler） | 用 `--no-amp` 仅排查 NaN |
| Optimizer | AdamW, lr=1e-5, wd=1e-4 | 见下方 lr 说明 |
| Scheduler | screening: linear warmup 1000 step → constant | strong run 见 §1.2.2 |
| Seed | 42 | `torch.manual_seed(42)`；DataLoader `worker_init_fn` 也固定 |
| `num_workers` | 4 | |

**Effective batch 必须固定为 64**（loss 曲线可比），但 micro-batch 按各实验显存裕度选。A0 的 `batch_size=96` 已在 VAE encoder attention 阶段 OOM，后续不作为默认配置；显存优先保证稳定完成 50 epoch，而不是追求单次 micro-batch 最大化：

| 实验 | 推荐 `batch_size` | `grad_accum_steps` | 备注 |
|---|---|---|---|
| A0 | 64 | 1 | 已验证可稳定跑完 50 epoch；bs96 有 OOM 风险 |
| B1 / B2 | 64 | 1 | DR + ControlNet；DR feature 接入生成主干；bs64 已验证可跑 |
| C1 / C2 / C3 | 64 | 1 | CBCT latent adapter + ControlNet/PACA；PACA attention 显存修复后 bs64 可跑 |
| D1（base_ch=64） | 64 | 1 | screening/diagnostic only |
| D1 strong（base_ch=128） | smoke 后确定 | smoke 后确定 | 优先保证 effective batch=64；OOM 则用 grad accumulation |
| D1 strong（base_ch=256） | smoke 后确定 | smoke 后确定 | 先做 smoke；能稳定训练则直接作为主提交候选 |

每个 run 必须把实际 `batch_size` × `grad_accum_steps` 写到 wandb config，便于审计。

**lr 说明**：旧版 5e-6 在 5 epoch smoke test 上 loss 能下降，但 50 epoch screening 内可能不足以让模块差异显现。screening 使用 1e-5，配 1000 step linear warmup 缓解前期不稳定。strong run 不再沿用 screening 的保守设置，见 §1.2.2。

> D1 strong run 允许使用更大 base_channels、更长训练、EMA 和 Stable Diffusion/LDM 风格学习率调度；所有变更必须在 wandb config 和 run note 中显式记录。

### 1.2.1 正则化与 EMA 策略（2026-05-04 更新）

- Screening 阶段默认 `dropout_rate=0.0`。`dropout_rate=0.1 + EMA` 的 A0 诊断 run 在 epoch 23 因 OOM 中断，且早期 decoded 指标弱于无 dropout A0；因此 dropout 不进入默认实验矩阵。
- EMA 代码已可用，但在 50 epoch screening 中不作为唯一模型选择依据。若开启 EMA，必须同时记录 raw 与 EMA 的 decoded 指标，或在 run note 中说明只用 EMA 评估的限制。
- D1 strong run **默认开启 EMA**。验证、fixed val 可视化、decoded MAE/PSNR/SSIM、best checkpoint 选择默认使用 EMA 权重；raw 权重仍保存，作为排查 EMA 滞后的辅助产物。L0 后续重训时使用同一策略。

### 1.2.2 Strong Run 训练策略（2026-05-05 更新）

strong run 的目标是拿到可提交候选，不再是低成本消融。当前优先训练 D1；L0 需要在当前数据集上重新训练，后置到 D1 有可用结果之后。D1 strong 使用以下训练策略：

| 项 | 默认 |
|---|---|
| `epochs` | 300–400；先设 400，early stopping 40 |
| `base_channels` | 优先 smoke `256`；若 D1-256 OOM 或 epoch time 不可接受，则降到 `128` |
| EMA | on，`ema_decay=0.9995` 优先；若 D1 代码只支持 0.999，需在 run note 写明 |
| LR | 优先复用 legacy 成功设置；初始建议 `1e-4`，若 D1 smoke 发散再降到 `5e-5` |
| LR schedule | Stable Diffusion/LDM 风格：linear warmup → constant LR；默认 warmup 10k steps，若总 step 较少则用总 step 的 3–5% |
| AMP | on |
| dropout | 默认 0.0；不与 strong 首跑绑定 |
| latent mode | `mu` |
| sampler | CBCT-init, `t_start=300`, DDIM 50 steps |

优先级规则：

- 若 D1-256 能跑，先完成 `D1-strong-bc256`，作为主提交候选。
- 若 D1-256 不能稳定跑，则跑 `D1-strong-bc128`。
- L0 不阻塞 D1。等 D1 有可用结果后，再用同一 manifest / VAE / fixed val / decoded metric 重训 L0，形成公平对照。
- 旧 legacy 结果来自另一数据集，只能作为“证明 concat 能 work”的历史参考，不能作为当前 D1 的正式对照。
- 不建议把 DR、PACA、dropout、latent sample 同时并入 D1-strong；这些不属于当前已验证有效的主变量。

### 1.3 WandB 与产物命名

- Project：`cbct2sct_IBA`
- Run group：`phase1-matrix-2026-05`
- Run name：`{exp_id}-bc{base_channels}-ep{epochs}-s{seed}`，例如 `A0-bc64-ep50-s42`；strong run 使用 `D1-strong-bc256-ep400-s42`，后续 L0 重训使用 `L0-strong-bc256-ep400-s42`
- Tags：必填 `exp_id`、`stage`（smoke/screen/strong）、`commit`（git short SHA）
- 保存目录：`checkpoints/phase1_matrix/{run_name}/`

### 1.4 前置条件（gating prerequisites）

执行 A0 / B / C / D1 之前**必须先合并**以下能力（见 §九）：

- `train_concat_paca.py` 增加 `--use-dr / --use-controlnet / --control-source {dr,cbct_latent} / --controlnet-fusion {add,paca,both}` 模块开关
- ControlNet 输入源切换：`--control-source {dr, cbct_latent}`。`dr` 使用 DR 输出的 `base_channels×64×64` feature；`cbct_latent` 使用 VAE-encoded CBCT latent 经 `3→base_channels` 1×1 ZeroConv adapter 后的 feature
- Fixed val 可视化与 decoded sCT 指标（MAE / PSNR / SSIM，分 region 记录）
- 推理 sampler：DDIM 50 步实现

未合并这些能力之前，所有运行都只是 smoke test，不计入实验矩阵。

---

## 二、评估协议

所有实验共享同一套评估协议，否则 metric 不可比。

### 2.1 Fixed validation set

**fixed val cases** 选定后写入 `configs/fixed_val_cases.yaml`，本批实验全程不变：

- 每个 region 选 4 例（共 16 例 patient），按 manifest val split 的字母序前 4 位
- 每例固定 z 索引：mask 前景面积最大的 3 个 slice（共 48 张 slice）
- 用于：图像可视化（CBCT / GT CT / sCT / |error|）+ 定性比较
- Fixed val 推理使用固定 seed=0；默认从 CBCT latent 加噪到 `t_start=300` 后执行 DDIM 去噪，保证跨 run 可视化可比

### 2.2 完整 val 集指标

每个实验在完整 val（23 例）上跑指标，每个 epoch 末或 `--eval-every N` 步：

| 指标 | 空间 | 计算范围 | 备注 |
|---|---|---|---|
| `val/loss_diff` | latent 64×64 | latent_mask 内 | 训练目标本身 |
| `val/loss_dr` | pixel 256×256 | mask 内 | 仅启用 DR 时记录 |
| `val/mae_hu` | pixel，反归一化回 HU | mask 内 | **主要排序指标** |
| `val/psnr` | pixel，[-1,1]，data_range=2 | mask 内 | |
| `val/ssim` | pixel，[-1,1]，data_range=2 | mask 内 | window=11 |
| `val/mae_hu_{BB,AB,HN,TH}` | 同上 | 分 region | 监控部位间差异 |

- sCT 重建：`x_t=CBCT-init(t_start=300) → DDIM(50 步) → VAE.decode → clamp(-1,1) → (x+1)/2*(1500-(-1024))-1024 = HU`
- **MAE 必须反归一化回 HU**（量纲依赖，"差 30 HU"才有临床可解释性）
- **PSNR / SSIM 在哪个范围算结果完全相同**（只要 `data_range` 跟着改：[-1,1] → 2、[0,1] → 1、HU → 2524）。统一用 [-1,1] + `data_range=2`，简洁且避免读者误以为换空间数值就变
- SSIM 计算范围由"全图"改为 mask 内：保持与 MAE / PSNR 一致，避免 padding 区域（值=-1，恒等）人为抬高 SSIM
- val MAE 作为模型选择的主排序指标；fixed val 图像作为定性检查

### 2.3 资源指标

每个 run 必须额外记录到 wandb：`gpu_mem_max_gb`、`epoch_time_sec`、`step_time_ms`、可训练参数量。

---

## 三、实验矩阵

### A0：主 Baseline

```text
Concat latent + region embedding + mask-weighted diffusion loss
NO DR module、NO ControlNet、NO PACA
```

CLI 等价：

```text
--use-dr=False --use-controlnet=False
```

目的：

- 建立最小可靠 baseline；后续所有增强模块都必须和 A0 比较
- 确认仅靠 CBCT latent concat + region embedding 能否生成可用 sCT

当前状态（2026-05-05）：

- `A0-bs64-cbct300-bc64-ep50-s42` 已完成 50 epoch，作为当前 screening baseline：`val/mae_hu=156.69`、`val/psnr=20.68`、`val/ssim=0.817`。
- `A0-drop01-ema-bs96-cbct300-bc64-ep50-s42` 只作为诊断 run：epoch 23 后在 VAE encoder attention OOM，未完成 50 epoch；早期指标 `val/mae_hu=213.73`、`val/psnr=18.25`、`val/ssim=0.761`，不纳入正式矩阵排序。
- A0 暂不继续反复调参。下一步优先验证 ControlNet 条件路径是否能突破 A0 上限。

### B 系列：ControlNet 条件源 = DR feature

```text
B1 = A0 + DR + ControlNet residual add, γ=0.5
B2 = A0 + DR + ControlNet residual add, γ=1.0
```

`--use-dr=True --use-controlnet=True --control-source=dr --controlnet-fusion=add --gamma {0.5, 1.0}`

注意：DR 必须接入生成主干才有实验意义。这里 DR 的 `controlnet_input` 作为 ControlNet 条件 feature 进入 UNet；`pred_128 / pred_64` 通过 γ 加权的 pixel-space L1 进行辅助监督。

本组回答的问题：

- DR feature 作为 ControlNet 条件，是否比 A0 更好？
- 在 DR feature 路径成立时，γ=0.5 还是 γ=1.0 更合适？

判断（量化）：

- **B 优于 A0**：`val/mae_hu` 降低 ≥ 5 HU 或 `val/ssim` 提升 ≥ 0.01，且 fixed val 图像无明显伪影增加
- **B1 vs B2**：差距 < 上述阈值视为持平，优先选更稳定的 γ=0.5
- **B 全部劣于 A0**：DR source 不进入 D1

当前结果（2026-05-05）：

- `B1-bs64-bc64-g05-ep50-s42` 已完成 50 epoch：`val/loss_total=0.29625`（含 DR aux，不能和 A0/C 直接比总 loss）、`val/loss_diff=0.19801`、`val/loss_dr=0.19649`、`val/mae_hu=145.07`、`val/psnr=21.29`、`val/ssim=0.827`。相对 A0 提升明显，但没有超过 C1/C3 到足以抵消 DR aux 复杂度。
- `B2-bs64-bc64-g10-ep20-s42` 只跑 20 epoch 诊断：`val/loss_total=0.44179`、`val/loss_diff=0.21005`、`val/loss_dr=0.23174`、`val/mae_hu=188.96`、`val/psnr=19.30`、`val/ssim=0.789`。γ=1.0 明显压坏 decoded 质量，已否决，不继续补 50 epoch。
- 结论：DR source 不是当前 D1 默认路径。若以后重开 DR，应从 γ≤0.5 或更弱 aux schedule 开始，而不是 γ=1.0。

### C 系列：ControlNet 条件源 = CBCT latent adapter

C 不启用 DR，ControlNet 输入由 "VAE-encoded CBCT latent + 1×1 ZeroConv adapter" 提供（adapter 通道 `3→base_channels`，参数归入 ControlNet）。在此前提下枚举 fusion 方式：

```text
C1 = A0 + ControlNet (residual add only)
C2 = A0 + ControlNet (PACA only)
C3 = A0 + ControlNet (residual add + PACA)
```

`--use-dr=False --use-controlnet=True --control-source=cbct_latent --controlnet-fusion {add, paca, both}`

C1 / C2 / C3 **统一跑 50 epoch**，避免不公平比较（旧版给 C3 30 epoch 是错的）。

判断（量化）：

- C1 / C2 / C3 之间差距 < 5 HU MAE 视为持平，选最简单的（C1 < C2 < C3）
- C 全部劣于 A0：ControlNet 路径不进入 D1
- C 中至少一个优于 A0 ≥ 5 HU MAE：保留最优融合方式

B 与 C 的核心对比：B 用 DR feature 作为 ControlNet 条件并附加 DR aux loss；C 用 CBCT latent adapter 作为 ControlNet 条件且无 aux loss。两组共同回答 "ControlNet 条件源选哪个"，C 内部再回答 "fusion 选哪种"。

当前结果（2026-05-05）：

| 实验 | fusion | epoch | `val/mae_hu` | `val/psnr` | `val/ssim` | `epoch_time_sec` | `gpu_mem_max_gb` |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | none | 50 | 156.69 | 20.68 | 0.817 | 211.19 | 7.45 |
| C1 | add | 50 | 144.54 | 21.31 | 0.830 | 229.76 | 10.55 |
| C2 | paca | 50 | 149.92 | 21.07 | 0.819 | 263.03 | 12.22 |
| C3 | both | 50 | 143.90 | 21.37 | 0.827 | 264.99 | 12.24 |
| B1 | dr+add, γ=0.5 | 50 | 145.07 | 21.29 | 0.827 | 236.43 | 11.41 |
| B2 | dr+add, γ=1.0 | 20 | 188.96 | 19.30 | 0.789 | - | - |

分 region MAE（HU）：

| 实验 | AB | BB | HN | TH |
|---|---:|---:|---:|---:|
| A0 | 212.73 | 136.59 | 168.91 | 131.87 |
| C1 | 202.72 | 124.64 | 154.01 | 120.78 |
| C2 | 197.54 | 135.76 | 158.14 | 124.53 |
| C3 | 190.34 | 129.36 | 151.65 | 121.33 |
| B1 | 197.97 | 130.08 | 152.29 | 117.63 |
| B2 | 263.72 | 171.42 | 186.06 | 158.96 |

结论：

- C1/C2/C3 都优于 A0，说明 ControlNet 条件路径有效。
- C2（PACA only）比 A0 好，但整体弱于 C1/C3，且更慢；PACA-only 不进入 D1。
- C3 总 MAE 最低、PSNR 最高，AB/HN 最好；但相对 C1 只好 0.64 HU，低于 5 HU 有意义阈值，且更慢、结构更复杂、SSIM 略低。
- 按“复杂模型与简单模型持平时优先简单”的规则，D1 默认选择 C1（CBCT latent source + residual add）。

### D1：最终主模型

```text
D1 = A0 + best ControlNet source + best fusion + best γ if source=DR
```

当前 D1 推荐（2026-05-05）：

```text
D1-strong = C1 strong run
--control-source cbct_latent
--controlnet-fusion add
--use-controlnet
--no-use-dr
--base-channels 256   # 先 smoke；若不可接受则降到 128
--use-ema --ema-decay 0.9995
--lr 1e-4             # 若 D1 smoke 发散则降到 5e-5
--lr-schedule sd-warmup-constant
--warmup-steps 10000  # 若总 step 较少则改为 3–5% total steps
--epochs 400
--sampler-init cbct --sampler-t-start 300 --sampler-alpha 1.0
```

选择依据：

- C1 相对 A0：MAE 156.69 → 144.54，SSIM 0.817 → 0.830，达到有效提升。
- C3 相对 C1：MAE 只多降 0.64 HU，未达到 5 HU 阈值；PACA+both 的额外复杂度和 epoch time 暂不划算。
- B1 相对 C1/C3：decoded 指标接近但无显著优势，还引入 DR aux loss 与条件源复杂度；B2 γ=1.0 明显失败。
- 因此 D1 默认不启用 DR，不启用 PACA；C3 可作为“若优先 AB/HN 最低 MAE 且接受复杂度”的可选替代，不作为默认主实验。
- D1 必须带 EMA 与学习率调度。C1@64/constant-like schedule 只用于 screening，不代表最终模型容量和训练策略。

若 D1-256 smoke 能稳定训练，则直接用 D1-256 进入 300–400 epoch；若 D1-256 OOM 或 epoch time 明显不可接受，则降到 D1-128。L0 重训后置，不阻塞 D1。

启动门控：

- B 和 C 系列已完成 screening；B2 因 γ=1.0 20 epoch 明显失败而提前否决
- 已确定 best 默认组合：CBCT latent source + residual add（C1）
- D1 训练脚本需要支持 strong run 所需的 EMA、Stable Diffusion/LDM 风格 warmup→constant 调度、EMA 权重评估/可视化
- 启动 D1 长训前在 wandb 写一条 note 说明组合选择依据、base_channels 选择和 lr schedule

### L0：Legacy Concat 后置公平对照

```text
L0 = legacy UNetConcatenation strong run
no ControlNet
no DR
no PACA
```

L0 的目的不是阻塞 D1，而是在 D1 有可用结果后，复现/更新 legacy 已经证明可行的强 concat baseline。当前 legacy 结果来自另一个数据集，不能直接作为 D1 的公平对照。若需要论文级或最终方案级结论，L0 必须在当前 manifest 上重新训练，并和 D1 使用同一份 VAE、同一套 fixed val、同一套 decoded metric，以及尽量一致的 strong run 训练策略：

```text
--base-channels 256   # 若 D1 只能跑 128，则补 L0-128
--use-ema --ema-decay 0.9995
--lr 1e-4             # 与 D1 对齐；必要时一起降到 5e-5
--lr-schedule sd-warmup-constant
--warmup-steps 10000
--epochs 400
--sampler-init cbct --sampler-t-start 300 --sampler-alpha 1.0
```

解释原则：

- `L0-256 vs D1-256` 是最干净的主结论。
- `L0-128 vs D1-128` 是 D1-256 跑不动时的公平替代。
- 旧 legacy 400 epoch 结果只作为历史参考；由于数据集不同，不能替代 L0-current retrain。

---

## 四、训练长度

### 4.1 Smoke test：1–5 epoch

每个新实验在正式 screening 前跑：

```text
--epochs 5 --max-train-batches 50 --max-val-batches 10
```

通过标准：无 OOM / NaN，train/val loss 正常记录，fixed val 图像成功上传 wandb。Smoke test 不计入实验矩阵。

### 4.2 Screening run：50 epoch

A0 / B1 / C1 / C2 / C3 一律 50 epoch，effective batch=64，seed=42。B2 原计划 50 epoch，但 20 epoch 时 decoded 指标已明显劣于 A0（MAE 188.96 vs 156.69），按提前停止规则终止并否决 γ=1.0。

提前停止条件（screening 阶段）：

- `val/mae_hu` 连续 15 epoch 无改善 → early stop
- 任意 epoch 出现 NaN / fixed val 图像明显发散 → 立即停
- 30 epoch 时 `val/mae_hu` 仍劣于 A0 同期 ≥ 10 HU → 提前停（节省算力）

### 4.3 Strong run：300–400 epoch

仅给：

- D1-strong（必跑）：300–400 epoch，默认 400，early stopping 40
- L0-strong（后置）：300–400 epoch；仅当需要当前数据集上的公平 concat 对照时启动
- early_stopping = 40

> D1 vs A0@50 不是公平对比；旧 legacy 又来自另一个数据集。当前优先拿到 D1-strong 主提交候选，L0-current retrain 后置为公平对照。

---

## 五、推荐执行顺序

```text
Step 0  合并 §1.4 前置 PR                         [done]
Step 1  A0     smoke 5 ep  →  screen 50 ep        [done: use A0-bs64-cbct300-bc64-ep50-s42]
Step 2  C1     smoke 5 ep  →  screen 50 ep        [done: recommended D1 default]
        C2     smoke 5 ep  →  screen 50 ep        [done: paca-only rejected]
        C3     smoke 5 ep  →  screen 50 ep        [done: optional alternative, not default]
Step 3  B1     smoke 5 ep  →  screen 50 ep        [done: useful but not default]
        B2     smoke 5 ep  →  screen 20 ep        [done: gamma=1.0 rejected]
Step 4  确定 best ControlNet source / fusion / gamma [done: C1, cbct_latent + add]
Step 5  补齐 D1 strong scheduler：SD/LDM warmup→constant [done]
Step 6  D1-256 smoke：显存、epoch time、loss/decoded sanity [next]
Step 7  若 D1-256 可跑：启动 D1-strong-bc256
        若 D1-256 不可跑：启动 D1-strong-bc128
Step 8  D1 有可用结果后，再决定是否重训 L0-current 作公平对照
```

当前不再建议补 `DR + PACA/both` 交叉点：C3 相对 C1 未达到 5 HU 阈值，B1 相对 C1/C3 无显著优势，B2 已失败。下一步应把算力用于 D1-strong。

D1-strong 启动建议：

```text
--use-controlnet --control-source cbct_latent --controlnet-fusion add
--base-channels 256
--use-ema --ema-decay 0.9995
--lr 1e-4 --lr-schedule sd-warmup-constant --warmup-steps 10000
--dropout-rate 0.0
--epochs 400
--sampler-init cbct --sampler-t-start 300 --sampler-alpha 1.0
```

先做 D1-256 smoke。若 OOM 或 epoch time 不可接受，降到 D1-128；若 D1-256 前 20–50 epoch 明显劣于 C1 screening 同期，先停下来复查 run config、固定 val 推理、VAE checkpoint、EMA 评估和 sampler 参数。

---

## 六、模型选择规则

按以下优先级，前面的判定即生效，不再看后面：

1. fixed val 图像：床板/固定装置误生成、明显伪影、边界错位等定性问题——出现即否决
2. `val/mae_hu`（整体 + 4 个 region 都看）：差异 ≥ 5 HU 视为有意义
3. `val/ssim`：差异 ≥ 0.01 视为有意义
4. `val/loss_diff`：仅作训练稳定性参考，**不作为模型间排序依据**
5. 训练稳定性：无 NaN / OOM / 周期性发散
6. 资源开销：epoch time ≤ 1.5× A0 视为可接受

冲突处理：

- 整体指标更好但某个 region（特别是 AB / TH）显著变差——记录但不直接采纳，需在 D1 长训中复查
- 简单模型与复杂模型持平时，**优先简单**
- 训练不稳定的"偶然最低 val_loss"不计

---

## 七、产物清单（每个 run 必须产出）

```text
checkpoints/phase1_matrix/{run_name}/
  ├── unet_full.pth                 # 当前 raw UNet 权重
  ├── unet_ema.pth                  # 仅 --use-ema 时保存
  ├── unet_ema_state.pth            # 仅 --use-ema 时保存，供续训
  ├── paca_layers.pth               # PACA/adapter 相关权重
  ├── controlnet_best.pth           # 仅 C / D1，若当前实现拆分保存
  └── dr_best.pth                   # 仅 B / D1，若当前实现拆分保存
```

wandb 指标见 §2.2 / §2.3，所有 run 必须完整。当前 fixed val 图像、loss 曲线和 decoded metrics 以 wandb 为主；若后续需要完全离线审计，再补 `train_config.yaml` / `metrics.csv` 文件落盘。

---

## 八、已知简化假设

明确记录，避免后续重复纠结：

1. **VAE 不重训**：当前 `vae_best.pth` 基于旧 train=199 split 训练；val 23 例不变，val 指标可横向比较。Phase 1 完成前不重训。
2. **Latent 用 mu，不用 sample**：节省一次随机采样，且 mu 在 reconstruction 任务上更稳定，不做消融。
3. **Region embedding 始终开启**：不做消融，相关价值参考已有 prior 工作。
4. **不全量展开 `control-source × fusion × gamma`**：B 固定 fusion=residual add，C 固定无 DR 后枚举 fusion；γ 仅在 B 内部消融。不跑完整笛卡尔积。
5. **DR source × {PACA, both} 不补测**：C3 比 C1 只好 0.64 HU，未达到 5 HU 阈值；B1 与 C1/C3 持平且更复杂，B2 已失败。当前没有足够收益支撑 `DR + PACA/both` 交叉点。
6. **不切 held-out test**：盲提交集（官方 val zip 42 例）由 D1 完成后单独写 inference 脚本。
7. **Phase 2 全局引导**：不在本矩阵内。仅当 D1 在 AB / TH 出现明显全局上下文缺失时再考虑。

---

## 九、Step 0 能力状态

按本 SPEC 执行所必需的代码改动：

- [x] `train_concat_paca.py` 增加 `--use-dr / --use-controlnet / --control-source {dr,cbct_latent} / --controlnet-fusion {add,paca,both}`，覆盖 A0 / B / C / D1
- [x] `--control-source=dr` 时，ControlNet 输入来自 DR 的 `base_channels×64×64` feature，并启用 `pred_128/pred_64` auxiliary loss
- [x] `--control-source=cbct_latent` 时，ControlNet 输入由 VAE-encoded CBCT latent 经 `3→base_channels` ZeroConv adapter 提供
- [x] Fixed val cases 配置：`configs/fixed_val_cases.yaml` + dataloader 支持
- [x] Decoded sCT 指标：MAE（HU）/ PSNR / SSIM，整体 + 分 region
- [x] DDIM 50 步 sampler，固定 seed；默认 CBCT-init `t_start=300`
- [x] WandB 记录 `gpu_mem_max_gb` / `epoch_time_sec` / `step_time_ms` / 可训练参数量
- [x] WandB run name / tag / group 按 §1.3 约定生成
- [x] `--seed` 参数 + DataLoader `worker_init_fn` 固定
- [x] D1 训练脚本支持 EMA 保存与 EMA 权重评估/可视化（`--use-ema`）
- [x] D1 训练脚本支持 Stable Diffusion/LDM 风格学习率调度参数（`--lr-schedule sd-warmup-constant --warmup-steps N`）
- [ ] L0-current retrain 与 D1 的 decoded metric / fixed val 可视化协议完全对齐（后置；若 legacy 脚本不一致，需要先补齐）

A/B/C screening 已完成。D1 strong 的 SD/LDM 风格 scheduler 已补齐；下一步做 D1-256 smoke，并按显存结果启动 D1-strong。
