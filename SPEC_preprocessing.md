# 数据预处理与训练方案

> 数据来源：synthRAD2023 Task2 (Brain) + synthRAD2025 Task2 (AB / HN / TH)

---

## 当前实现状态（2026-05-03）

- 预处理主线已实现于 `scripts/preprocess_synthrad.py`：SimpleITK 明确加载为 `(1, Z, H, W)`，MONAI `Compose` + `MapTransform` 负责同步 clip / crop / resize / pad / normalize。
- 本地 rawdata 训练 zip 已全量预处理完成：`data/preprocessed/` 9.2 GB，`data/manifest.csv` 241 例。
- 当前 zip 内训练病例数实测为 241 例，不是早期记录的 245 例：BB 60 / AB 53 / HN 65 / TH 63。
- `manifest.csv` 当前患者级划分：train 199 / val 23 / test 19。
- VAE 已完成全量训练验证：`checkpoints/vae/vae_best.pth` 在完整 val 集 2487 slices / 156 batches 上 `val_loss=0.252599`，无 NaN。
- VAE 架构形状已用真实 batch 验证：`(B,1,256,256) → (B,3,64,64) → (B,1,256,256)`。
- `scripts/train_vae_preprocessed.py` 已添加 `--eval-only`，可加载任意 VAE checkpoint 跑完整 val 并上传固定 val batch 可视化。
- Phase 2 全局引导已降级为可选增强；当前不阻塞 Phase 1 baseline。
- 真实小样本验证已完成：
  - `2ABD100` ConcatPACA 小模型 overfit 5 epoch：val loss 0.310 → 0.100。
  - `2ABD001` VAE 小样本 overfit 5 epoch：loss 0.350 → 0.083。
- `scripts/train_concat_paca.py` 已接入 WandB，默认上传；使用 `--no-wandb` 可关闭。Phase 1 小规模训练验证已完成，loss 可正常下降。

---

## 一、数据集分析

### 1.1 数据量

| 数据集 | 部位 | Spec 记录训练集 | 当前 zip 实测训练集 | 官方验证集（无 GT） |
|--------|------|----------------|--------------------|------------------|
| synthRAD2023 | Brain (BB) | 61 例 | **60 例** | 11 例 |
| synthRAD2025 | Abdomen (AB) | 54 例 | **53 例** | 9 例 |
| synthRAD2025 | Head & Neck (HN) | 66 例 | **65 例** | 11 例 |
| synthRAD2025 | Thorax (TH) | 64 例 | **63 例** | 11 例 |
| **合计** | | **245 例** | **241 例** | **42 例（提交用）** |

官方验证集仅含 CBCT + mask，无 CT ground truth，单独保留用于挑战赛提交。

### 1.2 文件格式

| 数据集 | 格式 | 每例文件 |
|--------|------|---------|
| 2023 | NIfTI `.nii.gz` | `ct.nii.gz` `cbct.nii.gz` `mask.nii.gz` |
| 2025 | MetaImage `.mha` | `ct.mha` `cbct.mha` `mask.mha` |

CT 与 CBCT **已预配准**（相同 shape / spacing / origin），无需额外配准。
Mask 为二值 body mask（前景约 42–50%）。

### 1.3 体积尺寸与 Spacing（实测）

| 部位 | Z 切片数 | 原始 XY (px) | Spacing (mm) |
|------|---------|-------------|-------------|
| Brain | ~162 | 192–239 | 1 × 1 × 1（各向同性）|
| AB | 69–99 | 308–497 | 1 × 1 × **3** |
| HN | 70–130 | 300–316 | 1 × 1 × **3** |
| TH | 70–137 | 465–499 | 1 × 1 × **3** |

2025 数据 Z 方向 3mm 厚层，2D 模型逐 slice 处理，**不做 Z 方向重采样**。

### 1.4 Mask Crop 后 XY 尺寸与有效分辨率（10 例采样）

| 部位 | crop 后 max(H,W) | scale → 256 | 有效分辨率 |
|------|----------------|-------------|----------|
| Brain | 222–244 | 1.05–1.15 | 1.0 mm/px |
| HN | 278–281 | 0.91–0.92 | 1.1 mm/px |
| AB | 290–419 | 0.61–0.88 | 1.6 mm/px |
| TH | 391–498 | 0.51–0.65 | 1.9 mm/px |

### 1.5 HU 强度分析（实测）

| 部位 | 模态 | p5 | p50 | p95 | p99 | max |
|------|------|-----|-----|-----|-----|-----|
| Brain | CT | – | – | – | – | 3000 |
| Brain | CBCT | – | – | – | – | 2000 |
| AB | CT | -1000 | -1000 | 41 | 84 | 3069 |
| AB | CBCT | -1000 | -1000 | -41 | 39 | 548 |
| TH | CT | -1000 | -1000 | 72 | 343 | 1530 |
| TH | CBCT | -1000 | -1000 | -138 | -4 | 1069 |

- 2023 float32，2025 int16 → 加载后统一转 float32
- CT 金属伪影可达 3000+ HU，clip 到 1500 消除

---

## 二、预处理方案

### 2.1 工具：MONAI

```bash
pip install "monai[itk,nibabel]"
```

MONAI 原生支持 `.nii.gz` 和 `.mha`；`Compose` + Dict 变换链保证 CT / CBCT / mask 空间变换严格同步。

### 2.2 处理流程

```
原始 volume (CT + CBCT + mask)
        │
        ▼
[Step 1] 加载 & 类型统一
         SimpleITK → (1, Z, H, W)，EnsureTyped → float32 / uint8
        │
        ▼
[Step 2] HU Clip：[-1024, 1500]
         clip=True，去金属伪影，保留全部软组织 / 骨
        │
        ▼
[Step 3] Mask-based 前景裁剪（margin=10px）
         CropForegroundd(source_key="mask")
        │
        ▼
[Step 4] 等比缩放 + 中心 Padding → 256×256
         scale = 256 / max(H_crop, W_crop)
         长边 resize 到 256，短边用 -1024 HU 中心填充
        │
        ▼
[Step 5] 归一化：[-1024, 1500] → [-1.0, 1.0]
        │
        ▼
[Step 6] 生成 cbct_global（Phase 2 备用，每例一个 volume）
         对裁剪前的原始 volume 直接 resize+pad → 256×256
         保存为独立 cbct_global.mha，不参与 Phase 1
        │
        ▼
[Step 7] 保存预处理后 volume 为 MHA
         ct_preprocessed.mha / cbct_preprocessed.mha / mask_preprocessed.mha
         写入更新后的 spacing / origin / direction

[Step 8] 保存 preprocess_metadata.json
         记录原始几何、crop bbox、resize scale、padding、输出几何
         供推理结果反向映射回原始网格
```

**输出格式选择 MHA（非 NPY slice）的原因：**
- 保留 spacing / origin / direction，可直接用 3D Slicer / ITK-SNAP 验证预处理质量
- 与 2025 输入格式一致，整个流程格式统一
- 每例仅 3 个文件（vs ~90 个 NPY），目录干净
- 本机 RAM 1007 GB，全量 volume 缓存（~12GB）完全可行，I/O 零代价

### 2.3 Step 4：等比缩放 + 中心 Padding

```
示例：crop 后 H=288, W=390
  scale = 256 / max(288, 390) = 0.656
  → resize 到 H'=189, W'=256
  → H 方向 padding 67px（上 33 / 下 34），值 = -1024 HU（空气）
  → 最终 256×256，与真实空气像素完全一致
```

各部位最大 padding：Brain ≤36px / HN ≤24px / AB ≤100px / TH ≤125px。

```python
from monai.transforms import MapTransform, Resized, SpatialPadd

class ResizeWithAspectRatioAndPadd(MapTransform):
    """等比缩放长边到 target_size，短边中心 padding 到 target_size。"""

    def __init__(self, keys, target_size=256, pad_value=-1.0, mask_key="mask"):
        super().__init__(keys)
        self.target = target_size
        self.pad_value = pad_value
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        H, W = d[self.keys[0]].shape[-2], d[self.keys[0]].shape[-1]
        scale = self.target / max(H, W)
        new_H, new_W = round(H * scale), round(W * scale)

        img_keys  = [k for k in self.keys if k != self.mask_key]
        mask_keys = [k for k in self.keys if k == self.mask_key]

        if img_keys:
            d = Resized(keys=img_keys, spatial_size=(-1, new_H, new_W),
                        mode="bilinear", anti_aliasing=(scale < 1.0))(d)
        if mask_keys:
            d = Resized(keys=mask_keys, spatial_size=(-1, new_H, new_W),
                        mode="nearest")(d)
        if img_keys:
            d = SpatialPadd(keys=img_keys,
                            spatial_size=(-1, self.target, self.target),
                            mode="constant", constant_values=self.pad_value)(d)
        if mask_keys:
            d = SpatialPadd(keys=mask_keys,
                            spatial_size=(-1, self.target, self.target),
                            mode="constant", constant_values=0)(d)
        return d
```

### 2.4 当前 MONAI Pipeline

```python
from monai.transforms import Compose, EnsureTyped, ScaleIntensityRanged, MapTransform, Resized, SpatialPadd

CLIP_MIN, CLIP_MAX = -1024, 1500
TARGET_SIZE = 256

preprocess = Compose([
    LoadSITKArrayd(keys=["ct", "cbct", "mask"]),      # 保持 (1, Z, H, W)
    EnsureTyped(keys=["ct", "cbct"], dtype=torch.float32),
    EnsureTyped(keys=["mask"], dtype=torch.uint8),
    ScaleIntensityRanged(                           # Step 2: HU clip
        keys=["ct", "cbct"],
        a_min=CLIP_MIN, a_max=CLIP_MAX,
        b_min=CLIP_MIN, b_max=CLIP_MAX, clip=True,
    ),
    MaskForegroundCropd(keys=["ct", "cbct", "mask"], source_key="mask", margin=10),
    ResizeWithAspectRatioAndPadd(                   # Step 4: resize + pad
        keys=["ct", "cbct", "mask"],
        target_size=TARGET_SIZE,
        pad_value=float(CLIP_MIN), mask_key="mask",
    ),
    ScaleIntensityRanged(                          # Step 5: normalize
        keys=["ct", "cbct"],
        a_min=CLIP_MIN, a_max=CLIP_MAX,
        b_min=-1.0, b_max=1.0, clip=True,
    ),
    BinarizeMaskd(keys=["mask"]),
])
```

保存：

```python
import SimpleITK as sitk

def save_preprocessed(data, out_dir, pid):
    for key in ["ct", "cbct", "mask"]:
        arr = data[key][0].numpy()          # (Z, 256, 256)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(updated_spacing)
        img.SetOrigin(updated_origin)
        img.SetDirection(original_direction)
        sitk.WriteImage(img, f"{out_dir}/{pid}/{key}_preprocessed.mha")
```

反向映射：

```python
restore_preprocessed_to_original(arr, preprocess_meta)
```

该函数会去除 center padding，将 256×256 resize 回 crop 尺寸，再按 `crop_bbox_xy` 放回原始 `(Z,H,W)` 网格。

### 2.5 manifest.csv 格式

每行一个 **患者**（volume 级别，不展开 slice）：

```
patient_id, region, split, ct_path, cbct_path, mask_path, cbct_global_path, preprocess_meta_path
2ABD100, AB, train, data/preprocessed/2ABD100/ct_preprocessed.mha, ..., data/preprocessed/2ABD100/preprocess_metadata.json
```

- `region`：BB / AB / HN / TH（用于 region embedding）
- `cbct_global_path`：整体缩放的全局 CBCT volume，当前仅作为可选增强备用
- `preprocess_meta_path`：反向映射和 QC 使用

### 2.6 数据划分（患者级）

当前本地 zip 实测划分：

| 部位 | 训练 | 验证 | 测试 | 合计 |
|------|------|------|------|------|
| Brain (BB) | 49 | 6 | 5 | 60 |
| AB | 44 | 5 | 4 | 53 |
| HN | 54 | 6 | 5 | 65 |
| TH | 52 | 6 | 5 | 63 |
| **合计** | **199** | **23** | **19** | **241** |

---

## 三、数据加载（SliceDataset）

### 3.1 策略

预处理输出为 MHA volume，训练时用 `utils/slice_dataset.py` 一次性缓存所有 volume 到内存，之后按 Z 轴提取切片：

- 全量 volume 缓存：~12 GB（1007 GB RAM 的 1.2%），加载一次后纯内存访问
- 每个 `__getitem__` 返回一个 256×256 slice + region_id
- 空气切片（mask 前景 < 100px）在构建 slice 索引时预先过滤

```python
import pandas as pd, torch
from torch.utils.data import Dataset

MASK_MIN_PIXELS = 100
REGION_TO_ID = {"BB": 0, "AB": 1, "HN": 2, "TH": 3}

class SliceDataset(Dataset):
    """从内存缓存 volume 展开为 2D slice。"""

    def __init__(self, manifest_csv, split, augmentation=None):
        df = pd.read_csv(manifest_csv)
        df = df[df["split"] == split].reset_index(drop=True)

        self._vols = []
        for _, row in df.iterrows():
            ct   = _load_mha(row["ct_path"])       # (Z, 256, 256)
            cbct = _load_mha(row["cbct_path"])
            mask = _load_mha(row["mask_path"])
            rid  = REGION_TO_ID[row["region"]]
            self._vols.append((ct, cbct, mask, rid))

        # 构建 (volume_idx, slice_z) 索引，过滤空气切片
        self.index = []
        for vi, (_, _, mask, _) in enumerate(self._vols):
            for z in range(mask.shape[0]):
                if mask[z].sum() >= MASK_MIN_PIXELS:
                    self.index.append((vi, z))
        self.augmentation = augmentation

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vi, z = self.index[idx]
        ct, cbct, mask, rid = self._vols[vi]
        ct   = torch.from_numpy(ct[z]).unsqueeze(0)
        cbct = torch.from_numpy(cbct[z]).unsqueeze(0)
        mask = torch.from_numpy(mask[z]).unsqueeze(0)
        region_id = torch.tensor(rid, dtype=torch.long)

        if self.augmentation:
            ct, cbct = self.augmentation(ct, cbct)

        return ct, cbct, mask, region_id
```

---

## 四、模型架构：UNetConcatControlPACA

### 4.1 三组件系统

```
CBCT (256×256, pixel space)
  │
  ├─→ [DegradationRemoval (DR)]
  │     ↓ controlnet_input (256ch, 64×64)
  │     ↓ pred_128, pred_64  ← 辅助监督 loss
  │
  ├─→ VAE.encode → cbct_z (3ch, 64×64)
  │
CT  → VAE.encode → ct_z → add noise → noisy_ct_z
  │
  ├─→ [ControlNet] (noisy_ct_z, controlnet_input, t)
  │     ↓ 8 down residuals + 1 middle residual（ZeroConv 初始化）
  │
  └─→ [UNetConcatControlPACA] (noisy_ct_z ⊕ cbct_z, t, ControlNet residuals)
        ↓ pred_noise
```

**总损失：**
```
L = MSE(pred_noise, noise) + γ × [L1(pred_128, gt_128) + L1(pred_64, gt_64)]
```

### 4.2 各组件说明

| 组件 | 参数量 | 作用 |
|------|--------|------|
| VAE | ~30M（**冻结**）| 图像 ↔ latent 压缩（仅用 CT 训练）|
| DegradationRemoval | ~0.3M | 像素空间 CBCT 降质建模，输出 64×64×256 特征 + 多尺度辅助预测 |
| ControlNet | ~184M | 复制 UNet 编码器结构，ZeroConv 注入结构化残差 |
| UNetConcatControlPACA | ~429M | 主去噪 UNet，concat 条件输入，PACA 层融合 ControlNet 残差 |
| **可训练合计** | **~613M** | |

**DR 模块的价值**：在 latent 空间操作之前，先在像素空间显式学习 CBCT 到 CT 的降质映射，相当于给 ControlNet 提供了一个"预清洁"的 CBCT 特征，对多部位的不同伪影模式尤为有效。

**PACA 层的作用**：上采样路径中对 ControlNet 残差做像素级注意力融合（Pixel-wise Attention for ControlNet-Assisted generation），优于简单加法注入。

### 4.3 显存分析（256×256 输入，AMP/BF16）

UNet / ControlNet 均在 **latent space（64×64）** 工作；DR 在像素空间但通道数少（16/32/96ch）。

> **注意**：旧文档写的 32×32 是错误的。VAE Encoder 只有 2 个 stride-2 下采样层（256→128→64），latent 实际是 **64×64×3**，不是 32×32×3。

| batch | 模型（weights+grads+Adam）| 激活值（64×64 实际）| 总估算 |
|-------|--------------------------|---------------------|--------|
| 4 | 7.4 GB | ~0.4 GB | ~7.8 GB |
| 8 | 7.4 GB | ~0.8 GB | ~8.2 GB |
| 16 | 7.4 GB | ~1.0 GB | ~8.4 GB |
| 32 | 7.4 GB | ~2.0 GB | ~9.4 GB |

**GPU 适配：**

| GPU | 显存 | 推荐 batch | 结论 |
|-----|------|-----------|------|
| RTX 3090 / 4090 | 24 GB | 16–32 | ✅ 舒适 |
| V100 / RTX 3080 Ti | 16 GB | 8–16 | ✅ 可行 |
| A100-40G | 40 GB | 32–64 | ✅ 充裕 |
| A100-80G / H100 | 80 GB | 64+ | ✅ 充裕 |

**实测修正**：旧估算明显偏乐观。RTX 4090 24GB 上，AMP 修复前 `base_channels=64` 且 `batch_size=8` 已在第一次 backward 发生 OOM（PyTorch reserved ~22.65GB，尝试再分配 4GB）。根因不是 DataLoader，而是训练循环只打印 `AMP Enabled: True`，没有实际使用 `autocast + GradScaler`。

当前已修复：

- `models/unetConcatControlPACA.py` 训练循环已真实启用 `torch.cuda.amp.autocast` 和 `GradScaler`。
- `scripts/train_concat_paca.py` 已新增 `--grad-accum-steps`，用 micro-batch 梯度累积实现更大等效 batch。
- `scripts/train_concat_paca.py` 已新增 `--no-amp`，用于必要时关闭 AMP 排查数值问题。

修复后短测：

- `base_channels=64, batch_size=2, grad_accum_steps=2` 跑通。
- `base_channels=64, batch_size=4, grad_accum_steps=2` 跑通。
- `base_channels=64, batch_size=8, grad_accum_steps=1` 单步跑通，说明原 OOM 主因已解除。

正式 Phase 1 长训建议：

- 起步使用 `base_channels=64`。
- 推荐 `batch_size=4, grad_accum_steps=2` 或 `batch_size=8, grad_accum_steps=1` 先跑 50 epoch。
- 若长训稳定再提高 batch 或尝试 `base_channels=128`。
- `base_channels=256` 原始设定在单张 4090 上仍需谨慎，建议先有 64/128 baseline 后再试。
- SPEC 中原参数量表仅保留为架构规模参考，不再作为显存承诺。

---

## 五、训练策略

### 5.1 Phase 1 Baseline

```
Phase 1：统一基线
  数据：BB + AB + HN + TH 当前 241 例，2D slice
  模型：DR + ControlNet + UNetConcatControlPACA（全量训练）
  条件：cbct_local（resize+pad → 256×256）
  附加：region embedding 注入 timestep embedding
  目标：各部位有效基线，验证 pipeline
```

Phase 1 是当前必须完成的 baseline。正式 VAE 和 Phase 1 长训稳定之前，不实现额外全局引导。

当前执行顺序：

1. 用 `vae_best.pth` 固定 VAE，先跑小规模 ConcatPACA Phase 1 训练，确认 loss 能下降、WandB 指标正常。**已完成**。
2. 小规模验证通过后，再启动完整 Phase 1 长训。
3. 若合成 CT 出现明显床板/固定装置伪影，再做 mask outside air 或 mask-weighted ablation；当前不先硬改预处理主线。

小规模 Phase 1 训练验证（2026-05-03）：

```bash
python scripts/train_concat_paca.py \
  --manifest data/manifest.csv \
  --vae-path checkpoints/vae/vae_best.pth \
  --save-dir checkpoints/concat_paca_phase1_smoke_b1 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 5 \
  --base-channels 64 \
  --lr 5e-6 \
  --gamma 1.0 \
  --max-train-batches 20 \
  --max-val-batches 5 \
  --wandb-project cbct2sct_IBA \
  --wandb-name concat-paca-phase1-smoke-b1-vae-best
```

结果：

- WandB run: `https://wandb.ai/SMU-BME/cbct2sct_IBA/runs/zk3xybff`
- Train: 每 epoch 20 batches；Val: 每 epoch 5 batches
- Train loss: `1.196547 → 1.121451`
- Val loss: `0.631105 → 0.602301`
- Val diffusion loss: `0.134271 → 0.124321`
- Val DR loss: `0.496834 → 0.477979`
- 结论：代码链路、VAE 冻结加载、mask-weighted diffusion loss、DR auxiliary loss、WandB 标量上传均正常。

失败配置记录：

```bash
--base-channels 64 --batch-size 8 --max-train-batches 20 --max-val-batches 5
```

该配置在 RTX 4090 24GB 第一次 backward OOM；正式训练不能直接按旧估算使用大 batch。

### 5.2 可选增强：Phase 2 全局引导

Phase 2 不是当前必做项，仅在 Phase 1 评估发现 AB / TH 明显缺少全局上下文时启用。

触发条件示例：

- AB / TH 相比 BB / HN 明显更差。
- 大范围体型、边界或 anatomical level 不稳定。
- 局部 crop 视野足够清晰，但整体位置先验不足。

若触发，再做：

```
Phase 2：全局引导 fine-tune（可选）
  额外输入：cbct_global（未 crop 的整体缩略图）
  架构改动：UNet init_conv 从 in×2 → in×3（local + global concat）
  训练：Phase 1 权重热启，lr 降 5–10 倍
```

`cbct_global.mha` 已在预处理阶段生成，保留备用；它不参与当前 Phase 1 baseline。

### 5.3 Region Embedding

```python
REGION_TO_ID = {"BB": 0, "AB": 1, "HN": 2, "TH": 3}

# UNet forward 中
t_emb = self.time_embedding(t)          # (B, D)
r_emb = self.region_embedding(region_id) # (B, D)，nn.Embedding(4, D)
cond  = t_emb + r_emb                   # 相加，维度不变
```

### 5.4 Mask-weighted Loss

Padding 区域（-1.0，空气）不应贡献 diffusion loss：

```python
# mask (B,1,256,256) → latent mask (B,1,64,64)
# kernel_size=4 because 256/64=4，不是 8（8 会给出 32×32）
latent_mask = F.avg_pool2d(mask.float(), kernel_size=4, stride=4) > 0.5
loss_diff = F.mse_loss(pred_noise * latent_mask, true_noise * latent_mask)
loss_dr   = degradation_loss(intermediate_preds, ct_img, mask)   # pixel-space mask 过滤
total     = loss_diff + gamma * loss_dr
```

### 5.5 WandB Logging

`scripts/train_concat_paca.py` 默认启用 WandB：

```bash
python scripts/train_concat_paca.py \
  --manifest data/manifest.csv \
  --vae-path checkpoints/vae/vae_best.pth \
  --save-dir checkpoints/concat_paca \
  --batch-size 16 \
  --epochs 300 \
  --lr 5e-6 \
  --wandb-project cbct2sct_IBA
```

可用 `--no-wandb` 关闭。短跑/调试可用：

```bash
--max-train-batches 2 --max-val-batches 2
```

当前记录指标：`train_loss`、`val_loss`、`train/loss_diff`、`train/loss_dr`、`val/loss_diff`、`val/loss_dr`、learning rate、batch 数等。

### 5.6 VAE Eval-only 验证

已提交 `--eval-only`，用于复核 VAE checkpoint，不进入训练循环：

```bash
python scripts/train_vae_preprocessed.py \
  --manifest data/manifest.csv \
  --save-dir checkpoints/vae_eval_best \
  --resume checkpoints/vae/vae_best.pth \
  --eval-only \
  --start-epoch 113 \
  --batch-size 16 \
  --num-workers 4 \
  --no-augment \
  --no-amp \
  --wandb-project cbct2sct_IBA \
  --wandb-name vae-best-val-eval \
  --vis-every 1 \
  --vis-num-samples 4
```

最近一次验证：

- WandB run: `https://wandb.ai/SMU-BME/cbct2sct_IBA/runs/dtdf0i1s`
- Val: 2487 slices / 156 batches
- `val_loss=0.252599`
- `val/l1=0.00627`，`val/mse=0.00018`，`val/ssim=0.00751`，`val/perceptual=1.32092`
- `val/kl=10822.86` 为未加权 KL；当前 `kl_weight=1e-5`，对总 loss 贡献约 0.108
- 可视化只上传固定 4 个 val 样本，每例 `original / reconstructed / absolute error map`，不是只验证 4 个样本。

---

## 六、代码改动清单（已完成）

以下三处已在 `d415fce` 及后续工作区修改中完成。

### 6.1 UNetConcatControlPACA：添加 Region Embedding

`models/unetConcatControlPACA.py` 已添加：

```python
# __init__ 中
self.region_embedding = nn.Embedding(4, time_emb_dim)

# forward 中（time_emb 计算之后）
t_emb = self.time_mlp(t)
r_emb = self.region_embedding(region_id)   # region_id: (B,) int
t_emb = t_emb + r_emb                      # 相加，形状不变
```

### 6.2 训练循环：更新 DataLoader 解包

`models/unetConcatControlPACA.py` 中 `train_unet_concat_control_paca` 已改为：

```python
# 旧（只解包 2 个值，会在 SliceDataset 的 4-tuple 上报错）
for i, (ct_img, cbct_img) in enumerate(train_loader):

# 改为
for i, (ct_img, cbct_img, mask, region_id) in enumerate(train_loader):
    ct_img    = ct_img.to(device)
    cbct_img  = cbct_img.to(device)
    mask      = mask.to(device)
    region_id = region_id.to(device)
```

然后把 `mask` 用于 5.3 的 mask-weighted loss，把 `region_id` 传入 `model(noisy, t, cbct_z, region_id=region_id)`。

### 6.3 接口一致性验证

pipeline 各环节形状已通过审计：

| 环节 | 形状 | 状态 |
|------|------|------|
| 预处理输出 | (1, Z, 256, 256) MHA | ✅ |
| SliceDataset 返回 | (B,1,256,256) × 2 + (B,1,256,256) + (B,) | ✅ |
| VAE encode | (B,3,64,64) latent | ✅（实测，非 32×32） |
| DR 输出 | (B,256,64,64) | ✅ |
| ControlNet init_conv 输入 | (B,3,64,64) = latent | ✅ |
| ControlNet h + cond 尺寸 | 64×64 == 64×64 | ✅ |
| PACA 空间维度匹配 | 各层 UNet↔ControlNet 对齐 | ✅ |
| Mask pool | kernel=4 → (B,1,64,64) | ✅（修正后）|
