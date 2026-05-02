# LDM-CBCT2SCT

基于潜在扩散模型（LDM）将锥形束CT（CBCT）转换为合成CT（sCT），用于放射治疗自适应治疗规划。

---

## 项目结构

```
ldm-cbct-sct-main/
├── models/
│   ├── vae.py                      # VAE 编解码器
│   ├── diffusion.py                # 前向扩散过程（加噪/调度）
│   ├── unet.py                     # 基础 UNet
│   ├── unetConditional.py          # Skip / Concat / CrossAttn 三种条件架构
│   ├── unetConditionalGuidance_.py # CFG 版本条件 UNet
│   ├── unetControlPACA.py          # ControlNet + PACA 架构
│   ├── unetConcatControlPACA.py    # Concat + PACA 混合架构
│   ├── controlnet.py               # ControlNet 模块
│   ├── degradationRemoval.py       # 降质去除模块
│   └── blocks.py                   # 基础模块（ResBlock, Attention, PACA 等）
├── utils/
│   ├── dataset.py                  # 数据集类（NPY / 配对 CBCT-CT）
│   ├── dataset_256.py              # 256×256 专用数据加载
│   ├── dataset_512.py              # 512×512 专用数据加载
│   ├── losses.py                   # 感知损失、SSIM 损失
│   ├── wandb_logger.py             # WandB 日志封装
│   └── constants.py                # 全局常量
├── scripts/
│   ├── train.py                    # VAE / UNet 通用训练入口
│   ├── train_experiment.py         # 多架构对比训练
│   ├── train_cfg.py                # CFG 模式训练
│   ├── train_cfg_focal.py          # CFG + Focal Loss 训练
│   ├── infer.py                    # DDIM 推理
│   ├── infer_cbct_init.py          # CBCT 初始化推理
│   ├── eval.py                     # 评估脚本（MAE/PSNR/SSIM）
│   ├── eval_experiment.py          # 批量实验评估
│   ├── npy_to_nifti.py             # NPY → NIfTI 转换
│   └── monitor_gpu_memory.py       # GPU 显存监控
├── configs/
│   └── base.py                     # 数据/模型/训练/推理配置类
├── rawdata/                        # 原始数据（zip）
├── checkpoints/                    # 训练权重
└── outputs/                        # 推理与评估结果
```

---

## 环境要求

- Python 3.8+，PyTorch 1.12+，CUDA GPU（显存 ≥ 24GB）

```bash
pip install torch torchvision numpy pandas scikit-image matplotlib tqdm wandb torch-ema SimpleITK
```

---

## 两阶段训练流程

### 阶段一：VAE 训练

VAE 将 256×256 图像压缩为 32×32×3 的潜在表示（压缩 64×），仅用 **CT 图像**训练。

```
输入 CT [1, 256, 256]
  → Encoder → μ, logσ² → 重参数化采样 z [3, 32, 32]
  → Decoder → 重建 CT [1, 256, 256]

损失：L1 + λ_perceptual·VGG + λ_ssim·SSIM + λ_kl·KL
```

```bash
python scripts/train.py \
    --stage vae \
    --manifest data/dataset/manifest.csv \
    --epochs 200 \
    --batch-size 32 \
    --lr 6.25e-6 \
    --save-dir checkpoints/vae
```

验收标准：重建 SSIM > 0.95，PSNR > 35 dB。

---

### 阶段二：条件 UNet 训练（潜在扩散）

冻结 VAE，在 latent 空间训练条件去噪 UNet。

```
CT → VAE.encode → ct_z
CBCT → VAE.encode → cbct_z  ← 作为条件

训练：对 ct_z 加随机噪声 → UNet(noisy_ct_z, cbct_z, t) → 预测噪声 ε
损失：MSE(pred_ε, true_ε)

推理：x_T ~ N(0,1) → DDIM 40步去噪（条件=cbct_z）→ VAE.decode → sCT
```

支持 Classifier-Free Guidance（CFG）：训练时以 10% 概率丢弃条件，推理时用 guidance scale 控制强度。

```bash
# 训练（CFG 模式）
python scripts/train_experiment.py \
    --arch concatenation \
    --manifest data/dataset/manifest.csv \
    --batch-size 16 \
    --base-channels 256 \
    --epochs 200 \
    --precision bf16 \
    --use-ema \
    --ema-decay 0.9999 \
    --use-cfg \
    --cfg-dropout 0.1
```

---

## 三种条件融合架构

| 架构 | 融合方式 | 参数量 | 文件 |
|------|---------|--------|------|
| **Concatenation** | CBCT latent 与噪声 latent 在输入层拼接（通道×2） | 293M | `unetConditional.py` |
| **Skip** | 独立 CBCT 编码器，上采样时逐层注入跳跃连接 | 447M | `unetConditional.py` |
| **CrossAttention** | 每个 block 中以 CBCT 特征为 K/V 做 cross-attention | 280M | `unetConditional.py` |

> **注意**：CrossAttention 要求 CBCT latent 与 CT latent 在同一语义空间。若 VAE 仅用 CT 训练，CBCT latent 未对齐，会导致 attention 映射错误。修复方案是用 CT+CBCT 联合训练 VAE。

---

## 推理

```bash
python scripts/infer.py \
    --vae checkpoints/vae/vae.pth \
    --unet checkpoints/unet/unet_best.pth \
    --manifest data/dataset/manifest.csv \
    --split test \
    --output outputs/inference \
    --ddim-steps 40 \
    --cfg-scale 7.5
```

输出为每张切片的 `.npy` 文件（HU 值）。可用 `scripts/npy_to_nifti.py` 转为 NIfTI。

---

## 评估

```bash
python scripts/eval.py \
    --pred outputs/inference \
    --gt data/dataset/CT \
    --manifest data/dataset/manifest.csv \
    --split test \
    --output outputs/evaluation/metrics.json
```

评估指标：

| 类别 | 指标 |
|------|------|
| 体素级 | MAE、RMSE、ME（HU） |
| 图像质量 | PSNR（dB）、SSIM |
| 分区域 | Full / Soft Tissue (±150 HU) / Low Density / High Density |

---

## 数据格式

预处理后的数据组织：

```
data/dataset/
├── CT/
│   └── {patient_id}/
│       └── {patient_id}_slice_{XXXX}.npy   # float32, HU 值
├── CBCT/
│   └── {patient_id}/
│       └── {patient_id}_slice_{XXXX}.npy
└── manifest.csv   # 列：ct_path, cbct_path, patient_id, slice_id, split
```

归一化：`normalized = HU / 1000.0`（`linear` 模式），或 `tanh(HU / 150)` （`tanh` 模式）。
