# 主实验矩阵 SPEC

> 目标：用可解释的实验顺序验证 Concat latent、DR、ControlNet、PACA 对 CBCT-to-sCT Phase 1 的边际贡献，最终确定一个可靠的 D1 主模型。
>
> 全部实验共享同一份预处理（见 `SPEC_pipeline.md`）和同一组 fixed val cases（见 §二）。每条实验必须可被另一个人按本 SPEC 重跑得到可比较的结果。

---

## 一、共同设置

### 1.1 数据与模型默认

- 数据：synthRAD2023 BB + synthRAD2025 AB / HN / TH
- Manifest：`data/manifest.csv`（patient-level：train 218 / val 23）
- VAE：`checkpoints/vae/vae_best.pth`，**冻结**，不在本批实验中重训
- Latent 模式：使用 VAE `mu`（不 reparameterize），不做消融
- Region embedding：始终开启，不做消融
- 主 loss：mask-weighted diffusion MSE（latent 空间，mask 由 256→64 avg-pool 得到）
- Inference sampler：DDIM 50 步，η=0（确定性）；评估默认用此设置（见 §二）

### 1.2 训练配置（screening 阶段必须一致）

| 项 | 值 | 说明 |
|---|---|---|
| `base_channels` | 64 | A/B/C 全部固定；D1 长训可单独调高 |
| **effective batch** | **8** | 跨实验固定的公平性约束 |
| `batch_size` × `grad_accum_steps` | 见下表 | 按各实验显存裕度选择，乘积=8 |
| AMP | on（autocast + GradScaler） | 用 `--no-amp` 仅排查 NaN |
| Optimizer | AdamW, lr=1e-5, wd=1e-4 | 见下方 lr 说明 |
| Scheduler | linear warmup 1000 step → constant | |
| Seed | 42 | `torch.manual_seed(42)`；DataLoader `worker_init_fn` 也固定 |
| `num_workers` | 4 | |

**Effective batch 必须固定为 8**（loss 曲线可比），但 micro-batch 按各实验显存裕度选：

| 实验 | 推荐 `batch_size` | `grad_accum_steps` | 备注 |
|---|---|---|---|
| A0 | 8 | 1 | 最小，无 ControlNet / DR |
| B1 / B2 | 8 | 1 | DR 模块仅 0.10M，几乎不增显存 |
| C1 / C2 / C3 | 4 | 2 | ControlNet 11.3M + PACA 反传激活，需要 accum |
| D1（base_ch=64） | 4 | 2 | 同 C |
| D1（如升 base_ch=128） | 2 | 4 | 长训前再决定是否升 |

每个 run 必须把实际 `batch_size` × `grad_accum_steps` 写到 wandb config，便于审计。

**lr 说明**：旧版 5e-6 在 5 epoch smoke test 上 loss 能下降，但 50 epoch screening 内可能不足以让模块差异显现。改用 1e-5（ControlNet 原论文同量级），配 1000 step linear warmup 缓解前期不稳定。A0 smoke test 必须先验证 1e-5 不发散；若发散则回退 5e-6 并在所有实验里同步回退。

> D1 长训允许在确定模块组合后调整 base_channels / batch / lr，但需要在 wandb 显式记录变更。

### 1.3 WandB 与产物命名

- Project：`cbct2sct_IBA`
- Run group：`phase1-matrix-2026-05`
- Run name：`{exp_id}-bc{base_channels}-ep{epochs}-s{seed}`，例如 `A0-bc64-ep50-s42`
- Tags：必填 `exp_id`、`stage`（smoke/screen/long）、`commit`（git short SHA）
- 保存目录：`checkpoints/phase1_matrix/{run_name}/`

### 1.4 前置条件（gating prerequisites）

执行 A0 / B / C / D1 之前**必须先合并**以下能力（见 §九）：

- `train_concat_paca.py` 增加 `--use-dr / --use-controlnet / --controlnet-fusion {add,paca,both}` 模块开关
- ControlNet 输入源切换：当 `--use-dr=False` 但 `--use-controlnet=True` 时，使用 VAE-encoded CBCT latent（3→256ch 1×1 conv adapter，ZeroConv 初始化）作为 ControlNet 输入
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
- Fixed val 推理使用 seed=0 的固定噪声起点（DDIM 起点 latent 由 `torch.Generator(device).manual_seed(0)` 生成）

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

- sCT 重建：`x_t=DDIM(50 步) → VAE.decode → clamp(-1,1) → (x+1)/2*(1500-(-1024))-1024 = HU`
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

### B 系列：DR 辅助监督

```text
B1 = A0 + DR (γ=0.5)
B2 = A0 + DR (γ=1.0)
```

`--use-dr=True --use-controlnet=False --gamma {0.5, 1.0}`

注意：A0 没有 ControlNet，DR 的 `controlnet_input` 输出在 B 系列中**没有下游消费者**，仅 `pred_128 / pred_64` 经 L1 与 GT CT 对齐，作为 CBCT→CT 像素空间辅助监督。这就是 B 想验证的——"在不引入 ControlNet 的情况下，给 encoder 加这条 pixel-space 辅助路径是否对 latent diffusion 有帮助"。

判断（量化）：

- **B 优于 A0**：`val/mae_hu` 降低 ≥ 5 HU 或 `val/ssim` 提升 ≥ 0.01，且 fixed val 图像无明显伪影增加
- **B1 vs B2**：差距 < 上述阈值视为持平，优先选更稳定的 γ=0.5
- **B 全部劣于 A0**：DR 不进入 D1，C 系列不再叠加 DR

### C 系列：ControlNet / PACA 融合

```text
C1 = A0 + ControlNet (residual add only)
C2 = A0 + ControlNet (PACA only)
C3 = A0 + ControlNet (residual add + PACA)
```

`--use-dr=False --use-controlnet=True --controlnet-fusion {add, paca, both}`

注意：C 系列不启用 DR，因此 ControlNet 输入由"VAE-encoded CBCT latent + 1×1 conv adapter (ZeroConv)"提供，不是 DR 输出。该 adapter 视为 ControlNet 的一部分，参数量归入 ControlNet。

C1 / C2 / C3 **统一跑 50 epoch**，避免不公平比较（旧版给 C3 30 epoch 是错的）。

判断（量化）：

- C1 / C2 / C3 之间差距 < 5 HU MAE 视为持平，选最简单的（C1 < C2 < C3）
- C 全部劣于 A0：ControlNet/PACA 不进入 D1
- C 中至少一个优于 A0 ≥ 5 HU MAE：保留最优融合方式

### D1：最终主模型

```text
D1 = A0 + best DR (from B) + best fusion (from C)
```

如果 B 全部劣于 A0：D1 退化为 A0 + best C
如果 C 全部劣于 A0：D1 退化为 A0 + best B
如果 B 和 C 都劣于 A0：D1 = A0 长训（200 epoch）即为最终模型

启动门控：

- B 和 C 系列全部完成 ≥ 50 epoch 且 fixed val 图像已上传
- 确定 best B（gamma）与 best C（fusion）
- 启动 D1 长训前在 wandb 写一条 note 说明组合选择依据

### 解耦假设

本矩阵假设 DR 与 ControlNet 融合方式之间无显著交互——即"先选最佳 γ、再选最佳 fusion、组合即最优"。我们不跑 B × C 的笛卡尔积。如果 D1 显著劣于预期（< A0 + max(best B, best C) 收益之和的一半），再回头补 1–2 个交叉点。

---

## 四、训练长度

### 4.1 Smoke test：1–5 epoch

每个新实验在正式 screening 前跑：

```text
--epochs 5 --max-train-batches 50 --max-val-batches 10
```

通过标准：无 OOM / NaN，train/val loss 正常记录，fixed val 图像成功上传 wandb。Smoke test 不计入实验矩阵。

### 4.2 Screening run：50 epoch

A0 / B1 / B2 / C1 / C2 / C3 一律 50 epoch，effective batch=8，seed=42。

提前停止条件（screening 阶段）：

- `val/mae_hu` 连续 15 epoch 无改善 → early stop
- 任意 epoch 出现 NaN / fixed val 图像明显发散 → 立即停
- 30 epoch 时 `val/mae_hu` 仍劣于 A0 同期 ≥ 10 HU → 提前停（节省算力）

### 4.3 Long run：200–300 epoch

仅给：

- D1（必跑）：200 epoch，必要时延伸到 300
- A0-long（**必跑**，作为同 budget 对照）：200 epoch
- early_stopping = 40

> 旧版把 A0-long 写成"可选"是错的——D1 vs A0@50 不是公平对比。A0-long 是 D1 的必备 reference。

---

## 五、推荐执行顺序

```text
Step 0  合并 §1.4 前置 PR
Step 1  A0     smoke 5 ep  →  screen 50 ep
Step 2  B1     smoke 5 ep  →  screen 50 ep
        B2     smoke 5 ep  →  screen 50 ep
Step 3  C1     smoke 5 ep  →  screen 50 ep
        C2     smoke 5 ep  →  screen 50 ep
        C3     smoke 5 ep  →  screen 50 ep
Step 4  确定 best B、best C，写 wandb note
Step 5  A0-long  200 ep    （与 D1 并行启动）
        D1       200–300 ep
Step 6  比较 D1 vs A0-long（200 ep 同 budget）
```

资源紧张时可先删 B2 或 C3（按 §三的判断阈值，二者更可能与同组其它 run 持平）。

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
  ├── unet_best.pth                 # 按 val/mae_hu 选 best
  ├── controlnet_best.pth           # 仅 C / D1
  ├── dr_best.pth                   # 仅 B / D1
  ├── train_config.yaml             # CLI args + git commit + seed
  ├── fixed_val_samples/{epoch}/    # CBCT/GT/sCT/error PNG
  └── metrics.csv                   # epoch, train_loss, val_loss, val/mae_hu, ...
```

wandb 指标见 §2.2 / §2.3，所有 run 必须完整。

---

## 八、已知简化假设

明确记录，避免后续重复纠结：

1. **VAE 不重训**：当前 `vae_best.pth` 基于旧 train=199 split 训练；val 23 例不变，val 指标可横向比较。Phase 1 完成前不重训。
2. **Latent 用 mu，不用 sample**：节省一次随机采样，且 mu 在 reconstruction 任务上更稳定，不做消融。
3. **Region embedding 始终开启**：不做消融，相关价值参考已有 prior 工作。
4. **B 与 C 解耦**：见 §三末尾，不跑笛卡尔积。
5. **不切 held-out test**：盲提交集（官方 val zip 42 例）由 D1 完成后单独写 inference 脚本。
6. **Phase 2 全局引导**：不在本矩阵内。仅当 D1 在 AB / TH 出现明显全局上下文缺失时再考虑。

---

## 九、待实现能力（Step 0 PR 范围）

按本 SPEC 执行所必需的代码改动：

- [ ] `train_concat_paca.py` 增加 `--use-dr / --use-controlnet / --controlnet-fusion`，三者组合覆盖 A0 / B / C / D1
- [ ] `--use-controlnet=True && --use-dr=False` 时，ControlNet 输入由 VAE-encoded CBCT latent 经 ZeroConv adapter 提供
- [ ] Fixed val cases 配置：`configs/fixed_val_cases.yaml` + dataloader 支持
- [ ] Decoded sCT 指标：MAE（HU）/ PSNR / SSIM，整体 + 分 region
- [ ] DDIM 50 步 sampler，固定噪声 seed
- [ ] WandB 记录 `gpu_mem_max_gb` / `epoch_time_sec` / `step_time_ms` / 可训练参数量
- [ ] WandB run name / tag / group 按 §1.3 约定生成
- [ ] `--seed` 参数 + DataLoader `worker_init_fn` 固定

完成后 §三的实验矩阵才可执行。
