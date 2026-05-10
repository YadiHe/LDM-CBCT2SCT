# SynthRAD2025 CBCT→sCT 实验记录

> 起点：2026-05-10（预处理 v2 切换日，旧 SPEC 全部归档）
> 主入口：`scripts/preprocess_synthrad_dataset.py` / `scripts/train_ct_vae.py` / `scripts/train_concat_paca.py`
> 数据：synthRAD2023 BB + synthRAD2025 AB / HN / TH，共 **241 例**，4 region（BB / AB / HN / TH）

---

## 0. TL;DR / 当前状态

- **预处理 v2 已切换**：HU clip 改为 `[-1024, 2000]`，新增 `SetAirBackgroundd`（mask 外强制为空气），归一化后 mask 外 ≈ -1.0。**所有旧 checkpoint 不能直接续训**（latent 分布变了）。
- **当前阻塞**：
  1. 重新生成预处理数据（覆盖 `data/preprocessed/` 并刷新 `data/manifest.csv`）
  2. 决定 VAE 是否重训（新数据分布与旧 VAE train set 不同）
  3. D1/E1 重训 baseline
- **已跑实验仅作历史参考**（基于预处理 v1：CLIP_MAX=1500、mask 外保留原 HU）。详见 §5。

---

## 1. 预处理 v2（当前）

### 1.1 入口与产物

```
scripts/preprocess_synthrad_dataset.py \
  --raw-dir  rawdata/ \
  --out-dir  data/preprocessed \
  --manifest data/manifest.csv
```

**每病例产物**（`data/preprocessed/{pid}/`）：

| 文件 | 形状 | 用途 |
|---|---|---|
| `ct_preprocessed.mha` | (Z, 256, 256) float32, [-1, 1] | 训练 GT |
| `cbct_preprocessed.mha` | (Z, 256, 256) float32, [-1, 1] | 训练条件（局部） |
| `mask_preprocessed.mha` | (Z, 256, 256) float32, {0, 1} | mask-weighted loss + 反归一化判定 |
| `cbct_global.mha` | (Z, 256, 256) float32, [-1, 1] | 全局 ControlNet 分支条件（不 crop body） |
| `preprocess_metadata.json` | — | 反向映射回原网格（推理/QC 用） |

`ct_preprocessed` / `cbct_preprocessed` / `mask_preprocessed` **完全对齐**（同一 bbox crop + 同一 resize + 同一 pad + 同 spacing/origin）。`cbct_global` 是另一个几何坐标系（不 crop），单独喂全局分支。

### 1.2 流程

```
load_patient_volumes (一次性读 ct/cbct/mask + 校验同一 voxel grid)
        │
        ▼
EnsureTyped (ct/cbct→float32, mask→uint8)
        │
        ▼
ScaleIntensityRanged  clip → [-1024, 2000]
        │
        ▼
MaskForegroundCropd   union body bbox + margin=10 px
        │
        ▼
SetAirBackgroundd     mask=0 区域强制 -1024 HU
        │
        ▼
ResizeWithAspectRatioAndPadd  长边→256 (bilinear/nearest), 短边中心 pad -1024
        │
        ▼
ScaleIntensityRanged  归一化 [-1024, 2000] → [-1, 1]
        │
        ▼
BinarizeMaskd         mask > 0.5 → {0, 1}
```

**Global 分支**：跳过 crop（直接对原始 volume 做 SetAirBg + resize+pad + 归一化），其余流程同上。

### 1.3 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| `CLIP_MIN` | -1024.0 | 空气 |
| `CLIP_MAX` | **2000.0** | 保留致密骨/牙齿，截断金属伪影（v1 是 1500） |
| `TARGET` | 256 | 输出 XY 尺寸 |
| `MARGIN` | 10 | mask bbox 外扩像素 |
| Resize 模式 | bilinear (img) / nearest (mask) | resize 后 1-2 像素边界存在 bilinear 平滑残留 |
| Pad 值 | -1024 (img) / 0 (mask) | |
| Z 重采样 | **不做** | 2D slice 模型逐层训练 |

### 1.4 与 v1 的差异

| 项 | v1（旧） | v2（当前） |
|---|---|---|
| `CLIP_MAX` | 1500 | **2000** |
| mask 外背景 | 保留原 HU（含床板/散射/伪影） | **强制 -1024 HU** |
| IO 次数/病例 | ~9 次 sitk.ReadImage | **3 次** |
| 几何校验 | 无 | `load_patient_volumes` 一次性校验 |
| 验证 | 无 | 第 1 例 assert `bg.max() ≤ -1+0.05` |

### 1.5 输出健全性检查

预处理 main 末尾对**第一例**做：
- 读回 `ct_preprocessed.mha` + `mask_preprocessed.mha`
- 取 mask=0 区域
- assert `max ≤ -1 + 0.05`（容许 bilinear 边界残留）

如果失败，整个 batch 终止 —— 比 v1 静默打印更安全。

### 1.6 manifest 格式

```
patient_id, region, split, ct_path, cbct_path, mask_path, cbct_global_path, preprocess_meta_path
```

`region ∈ {BB, AB, HN, TH}`，`split ∈ {train, val}`。所有路径是绝对路径。

---

## 2. 数据

### 2.1 数据集

| 数据集 | region | 训练 zip（含 CT GT） | 官方 val（无 GT） |
|---|---|---:|---:|
| synthRAD2023 Task2 | BB | 60 | 11 |
| synthRAD2025 Task2 | AB | 53 | 9 |
| synthRAD2025 Task2 | HN | 65 | 11 |
| synthRAD2025 Task2 | TH | 63 | 11 |
| **合计** |  | **241** | **42（提交用，不在 manifest）** |

CT 与 CBCT 已配准（同 shape/spacing/origin）。Mask 为二值 body mask（前景约 42–50%）。

### 2.2 划分（patient-level）

每个 region 字母序排序，`SPLIT_COUNTS` 切：

| region | 训练 | 验证 |
|---|---:|---:|
| BB | 54 | 6 |
| AB | 48 | 5 |
| HN | 59 | 6 |
| TH | 57 | 6 |
| **合计** | **218** | **23** |

不切 held-out test。挑战赛盲提交集（官方 val 42 例）由 Phase 1 完训后单独跑 inference。

### 2.3 体积统计

| region | Z | 原始 XY (px) | Spacing (mm) | crop 后 max(H,W) | →256 scale |
|---|---|---|---|---|---|
| BB | ~162 | 192–239 | 1×1×1 | 222–239 | 1.07–1.15 |
| HN | 70–130 | 300–316 | 1×1×3 | 278–281 | 0.91–0.92 |
| AB | 69–99 | 308–497 | 1×1×3 | 290–419 | 0.61–0.88 |
| TH | 70–137 | 465–499 | 1×1×3 | 391–498 | 0.51–0.65 |

2025 数据 Z 方向 3 mm 厚层。

---

## 3. 模型架构

### 3.1 总览

```
CBCT (1, 256, 256, pixel)
   │
   ├──→ [DR  optional]  → controlnet_input (256ch, 64×64) + 多尺度辅助预测
   │
   ├──→ VAE.encode      → cbct_z (3, 64×64)
   │
CT → VAE.encode → ct_z → +noise → noisy_ct_z
   │
   ├──→ [ControlNet]    → 8 down + 1 mid 残差（ZeroConv 初始化）
   │
   └──→ [UNetConcatControlPACA]
            input  : concat(noisy_ct_z, cbct_z) + region_emb + t_emb
            fusion : add | paca | both
            output : pred_noise (or v) (3, 64, 64)
```

VAE 仅用 CT 训练，inference 时也用同一 VAE encode 整个 CBCT 作为 latent 条件。

### 3.2 组件参数（实测，按 base_channels）

| 组件 | base=64 | base=128 | base=256 | 说明 |
|---|---:|---:|---:|---|
| VAE | 9.5M（冻结） | 9.5M | 9.5M | (B,1,256,256)↔(B,3,64,64) |
| DR | 0.10M | 0.16M | 0.27M | 像素空间 CBCT→CT 降质映射 + 多尺度辅助 |
| ControlNet | 11.3M | 45.0M | 179.9M | 复制 UNet encoder + ZeroConv |
| UNet (主) | 19.9M | 79.4M | 317.2M | 含 PACA 上采样路径 |
| **可训练合计** | **~31M** | **~125M** | **~497M** | |

> **注意**：`fusion=add` 下 PACA 参数（约 24M @ bc256）不参与 forward，活跃参数 ~497M；`fusion=paca/both` 才激活。

### 3.3 关键设计

- **Region embedding**：`nn.Embedding(4, time_emb_dim)` → 加到 `t_emb` 作为条件
- **PACA**（Pixel-wise Attention for ControlNet-Assisted）：上采样路径对 ControlNet 残差做像素级注意力
- **Mask-weighted latent loss**：`F.avg_pool2d(mask, kernel_size=4)` 把 256×256 mask 压到 64×64，作为 diffusion loss 权重
- **DR 辅助 loss**：`L = MSE(noise) + γ × [L1(pred_128, gt_128) + L1(pred_64, gt_64)]`
- **Latent 模式**：`mu`（不 reparameterize），latent_scale=1.3995

---

## 4. 评估协议

### 4.1 主排序指标

**`val/mae_hu`**（mask 内，反归一化回 HU 空间）。
`val/loss_diff`（latent 噪声预测 loss）**不参与排序**，仅看稳定性。

### 4.2 完整 val 指标（每 `eval_every` epoch）

| 指标 | 范围 | 说明 |
|---|---|---|
| `val/mae_hu` | mask 内 HU | **主排序** |
| `val/psnr` | mask 内，data_range=2 | [-1, 1] 空间 |
| `val/ssim` | mask 内，data_range=2，window=11 | [-1, 1] 空间 |
| `val/mae_hu_{AB,BB,HN,TH}` | 同上 | 分 region 监控 |
| `val/loss_diff` | latent | 稳定性参考 |

sCT 重建路径：`CBCT latent concat → UNet+ControlNet 去噪 → VAE decode → 反归一化 HU`。

### 4.3 Sampler 协议

DDIM 离散化误差 ∝ 1/K，每步覆盖 timestep = `t_start / K`：

| Sampler | 每步覆盖 | 用途 |
|---|---:|---|
| `cbct / t300 / DDIM50` | 6 | D1 系（CBCT-init） |
| `noise / t999 / DDIM100` | 10 | E1 系（标准全去噪） |
| `noise / t999 / DDIM200` | 5 | 官方提交 |

**跨 run 比 MAE 必须先在 best EMA 上跑 sampler sweep 对齐**：
```
init ∈ {cbct, noise} × t_start ∈ {200,300,500,700,999} × ddim_steps ∈ {50,100,200} × α ∈ {0.7, 1.0}
```

### 4.4 Fixed val 定性检查

- 配置：`configs/fixed_val_cases.yaml`
- 每 region 4 例 × 3 slice = 16 例 / 48 slices，固定 z 索引（mask 前景面积最大）
- 推理：固定 seed=0，run 协议同 train-time
- 用途：跨 run 视觉对比、伪影排查；**不参与排序**

### 4.5 模型选择规则（优先级）

1. 图像质量：床板/固定装置误生成、明显伪影、边界错位 → 出现即否决
2. `val/mae_hu`（整体 + 4 region），差异 ≥ 5 HU 视为有意义
3. `val/ssim`，差异 ≥ 0.01 视为有意义
4. 训练稳定性：无周期性 NaN / OOM
5. 资源：epoch time ≤ 1.5× baseline

简单模型与复杂模型持平 → 优先简单。

---

## 5. 已跑实验（基于预处理 v1，作为历史参考）

> ⚠ 以下结果均基于 v1（CLIP_MAX=1500 + mask 外保留原 HU），**不可直接外推到 v2**。Latent 分布变化后 MAE 数字会偏移，需要在 v2 数据上重做 baseline 才能形成可比序列。

### 5.1 VAE 重建基线（2026-05-07）

| 指标 | 整体 | AB | BB | HN | TH |
|---|---:|---:|---:|---:|---:|
| `mae_hu_vae` | **24.99 HU** | 20.98 | 29.11 | 22.10 | 22.80 |
| `psnr_vae` | 35.19 dB | — | — | — | — |
| `ssim_vae` | 0.9778 | — | — | — | — |

WandB run: `dtdf0i1s` · val 23 例 / 2487 slices · `val_loss=0.252599`
checkpoint: `checkpoints/vae/vae_best.pth`（基于旧 train=199 split）

**结论**：v1 上 VAE 不是瓶颈（理论下界 ~25 HU vs D1 pilot 101 HU，gap=76 HU）。**v2 上是否仍成立未验证**。

### 5.2 Screening 结果（A0/B/C，50 epoch）

| 实验 | 架构 | epoch | MAE | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| A0 | concat only | 50 | 156.69 | 20.68 | 0.817 |
| C1 | +ControlNet add | 50 | 144.54 | 21.31 | 0.830 |
| C2 | +ControlNet PACA only | 50 | 149.92 | 21.07 | 0.819 |
| C3 | +ControlNet add+PACA | 50 | 143.90 | 21.37 | 0.827 |
| B1 | +DR+ControlNet γ=0.5 | 50 | 145.07 | 21.29 | 0.827 |
| B2 | +DR+ControlNet γ=1.0 | 20 | 188.96 | 19.30 | 0.789 |

> **方法学注意**：50-epoch latent loss 与最终 decoded HU 排序相关性弱；screening 只用于排除明显无效配置（如 B2 γ=1.0 否决）。
> 选定架构：**C1**（C3 仅领先 0.64 HU，简单优先）→ D1。

### 5.3 D1-strong-bc256 pilot

WandB `ab94l2m1`，bs24 → ep22 NaN → 从 ep22 EMA 恢复 bs20+lr5e-5 续 28 ep，best EMA 在恢复 ep18：

| 指标 | 数值 |
|---|---:|
| `val/mae_hu` | 101.01 |
| `val/psnr` | 23.39 dB |
| `val/ssim` | 0.8796 |
| MAE AB / BB / HN / TH | 128.70 / 88.84 / 114.22 / 84.52 |

### 5.4 D1-bf16-drop01（继 pilot，2026-05-07 中断）

WandB `5qwexwip`，pilot ckpt 续训到 ep38 后中断：

| 指标 | 数值 |
|---|---:|
| `val/mae_hu` | 97.44 |
| `val/psnr` | 23.37 dB |
| `val/ssim` | 0.8811 |
| MAE AB / BB / HN / TH | 116.92 / 88.45 / 111.31 / 80.67 |

### 5.5 D1-l1-minsnr-cosine（已停止 ❌）

WandB `8ypydmdw`，从 D1-bf16-drop01 ckpt 切 L1+Min-SNR+cosine+t999/noise：

- ep38 decoded `val/mae_hu = 253.07 HU`（AB/BB/HN/TH = 376/195/252/272）
- **失败原因**：从 MSE-trained ckpt 直接切 L1+Min-SNR+t999/noise 的目标和 sampler 同时切换不兼容
- **结论**：L1/cosine/Min-SNR 组合必须从头训（即 E1）

### 5.6 E1-cosine-fullpaca（启动于 v1，启动方式 `scripts/launch_e1.sh`）

| 配置 | 值 |
|---|---|
| 架构 | UNet bc256 + DR + ControlNet (`control-source=dr`, `fusion=both`) |
| Noise schedule | cosine (iDDPM) |
| Prediction | v_prediction |
| Timestep sampling | logit_normal |
| Latent scale | 1.3995 |
| LR | 1e-4 cosine warmup 1000 step → 0.1× |
| Loss | L1 + Min-SNR γ=5 |
| Sampler (val) | noise / t999 / DDIM100 |
| AMP | bf16 |
| Batch | 24 |

> 切到 v2 后此 run 需要重启；v1 期间数据点保留作 noise schedule / loss 配置参考。

### 5.7 Pipeline 形状审计（仍适用于 v2）

| 环节 | 形状 |
|---|---|
| 预处理输出 | (1, Z, 256, 256) MHA |
| SliceDataset 返回 | (B,1,256,256) ×2 + (B,1,256,256) + (B,) |
| VAE encode | (B,3,64,64) |
| DR 输出 | (B, base_ch, 64, 64) |
| ControlNet h ↔ UNet h | 64×64 对齐 |
| Mask pool（kernel=4） | (B,1,64,64) |

---

## 6. 待跑实验

### 6.1 立即（解锁 v2）

| 步骤 | 内容 | 产物 |
|---|---|---|
| **P0** | 跑预处理 v2，覆盖 `data/preprocessed/` 并刷新 `data/manifest.csv` | 218+23 例 v2 数据 |
| **P1** | 在 v2 数据上 eval 旧 VAE，看 mae_hu_vae 是否仍 ≈ 25 HU | 决定是否 P2 |
| **P2** | 若 P1 退化 ≥ 10 HU：用 v2 train=218 重训 VAE | 新 `vae_best_v2.pth` |
| **P3** | 在 v2 数据 + (旧或新) VAE 上重做 D1 baseline 50 ep（screening 等价） | v2 baseline MAE |

### 6.2 D1 主线（v2 上重新校准）

D1 = `UNet bc256 + ControlNet(cbct_latent) + region_emb`，`fusion=add`，活跃 ~497M。

训练协议先回到稳态配置：

| 项 | 值 |
|---|---|
| `base_channels` | 256 |
| EMA | 0.9995 |
| AMP | bf16 |
| Optimizer | AdamW wd=1e-4 |
| LR schedule | `sd-warmup-constant`，warmup 1000，lr 3e-5 |
| Loss | MSE，mask-weighted |
| Dropout | 0.1 |
| Batch | 24（OOM 降 20） |
| Sampler (val) | cbct / t300 / DDIM50（与 v1 D1 协议一致便于对比，但 mae 不可直接换算） |
| `eval_every` | 10 |
| Seed | 42 |

### 6.3 消融（D1 收敛后）

Top-down 设计：以 D1-full 为基准，逐模块关闭，**其余配置完全一致**。

| 实验 | 关闭 | 回答的问题 | 优先级 |
|---|---|---|---|
| D1-full（基准） | — | 全模型表现 | ✓ |
| D1-no-controlnet | ControlNet（保留 concat + region） | ControlNet 边际贡献 | **1** |
| D1-no-region | region embedding | region embedding 边际贡献 | 2 |
| D1-baseline | 两者都关 | 联合贡献（≈ 重训预算的 A0） | 3 |

每个消融**从头训练**（不 fine-tune）。

### 6.4 E1 主线（SOTA 训练协议对照）

E1 配置见 §5.6（cosine + L1 + Min-SNR + v_prediction + noise/t999/DDIM100）。
**重启时机**：D1 在 v2 上跑通 ep30 之后再启动 E1，避免双线同时占卡且未对齐 baseline。

### 6.5 Joint → per-region fine-tune（提交策略）

挑战 SynthRAD 的主线采用两阶段：

1. **先训全局 joint model**：VAE-v2、D-v2、E-v2 都先用 AB/BB/HN/TH 全部训练集一起训练；D/E 保留 `region_emb`，得到公平的全局 baseline 和 per-region val 误差分布。
2. **再按部位微调**：从全局 D/E best checkpoint 出发，分别 fine-tune `AB-only` / `BB-only` / `HN-only` / `TH-only` 模型；LR 用全局训练的 0.1×–0.3×，短程训练，按对应 region val MAE 早停。
3. **提交候选**：推理时按 case region 路由到对应 fine-tuned 模型；同时保留全局 E 模型作为稳健 fallback。

原则：VAE 只训一个全局版本，避免不同 region VAE 造成 latent 分布不一致；per-region fine-tune 只用于修正解剖区域特异性偏差，不作为 D/E 公平对比的起点。

### 6.6 Sampler sweep（best EMA 后）

```
init ∈ {cbct, noise} × t_start ∈ {200,300,500,700,999} × ddim_steps ∈ {50,100,200} × α ∈ {0.7, 1.0}
```

重点对照：
- `t999/DDIM50` vs `t999/DDIM200` → 离散化误差大小
- `t999/DDIM200` vs `t300/DDIM50` → CBCT-init 是否仍有优势

### 6.7 决策树（D1 v2 baseline）

| 检查点 | MAE 阈值 | 行动 |
|---|---|---|
| ep10 | < 95 HU | 继续到 ep30 |
| ep10 | 95–110 HU | 继续观察到 ep30，并行准备 sampler sweep |
| ep10 | > 110 HU | 停训，排查 sampler/EMA/lr |
| ep30 | < 90 HU | 路径 A：继续到 ep100 |
| ep30 | 90–105 HU | 路径 B：平台期，停止长训 + sampler sweep + 小 LR fine-tune |
| ep30 | > 105 HU | 路径 C：转 MAISI POC（B3） |
| ep50 | < 80 HU | 继续到 ep100，目标落点 70–85 HU |
| ep50 | ≥ 90 HU | 启动 B2/B3 |

### 6.8 回退路径

| 路径 | 触发 | 工作量 | 预期 MAE |
|---|---|---|---|
| **B1** | ep50 卡 80–100 HU | ~1 天 | 加 perceptual loss + 强增强；70–85 HU |
| **B2** | ep50 卡 80–100 HU | ~1–2 天 | cosine + v_prediction 重训（≈ E1）；60–75 HU |
| **B3** | ep50 > 90 HU 或 B2 不足 | ~1–2 周 | **MAISI 3D-RFlow**；50–65 HU |

---

## 7. 产物清单与命名

```
checkpoints/phase1_matrix/{run_name}/
  ├── unet_full.pth          # raw UNet（续训用）
  ├── unet_ema.pth           # EMA（推理用）
  ├── unet_ema_state.pth     # EMA state（shadow + decay，续训 EMA 用）
  ├── controlnet_full.pth    # raw ControlNet（续训用）
  ├── controlnet.pth         # EMA ControlNet（推理用）
  ├── control_adapter_full.pth / control_adapter.pth
  ├── paca_layers.pth
  └── predictions/           # fixed val 输出
```

- WandB project：`cbct2sct_IBA`，group：`phase1-matrix-2026-05`
- Run name：`{exp_id}-bc{base_channels}-bs{batch}-ep{epochs}-s{seed}`
- Tags：`exp_id`、`stage` (smoke/screen/strong/long)、`commit` (git SHA)、`prep_v2`（标记 v2 预处理）

> **续训参数说明**：脚本无 `--resume`。需分别传 `--unet-path`（`unet_full.pth`，非 EMA）、`--ema-path`（`unet_ema_state.pth`）、`--controlnet-path`（`controlnet_full.pth`，非 EMA）、`--control-adapter-path`（`control_adapter_full.pth`）。
> `unet_ema.pth`/`controlnet.pth` 是 EMA 推理权重，**不用于续训**。

---

## 8. 已知局限

1. **预处理 v1 → v2 不向后兼容**：所有 v1 时期的 D1/VAE checkpoint 都不能直接续训 v2 数据；§5 所有结果仅作历史参考。
2. **VAE train split**：旧 VAE 基于 train=199；当前 split 是 218。Phase 1 内是否重训 VAE 待 P1 验证决定。
3. **PACA 死权重**：`fusion=add` 下 PACA 参数不参与 forward，仅占存储和 optimizer 状态。报数时使用活跃参数 ~497M。
4. **续训目标切换高风险**：v1 上已验证从 MSE ckpt 切 L1+Min-SNR+t999/noise 会崩；v2 重启后 D1/E1 严格分线，不混合协议续训。
5. **单 seed**：seed=42 单次。最终结论前补 seed=43 至少复现 D1 + 1 个消融。
6. **2D slice 架构**：损失 3D 上下文，slice 间一致性是 SynthRAD 体积级评分的潜在扣分项。如果 D1 卡 80+ HU，B3（MAISI 3D）是必要回退。
7. **官方 held-out**：盲提交集（官方 val 42 例）当前预处理脚本不读它，留作 Phase 1 完训后的单独 inference 步骤。
