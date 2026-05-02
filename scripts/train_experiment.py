#!/usr/bin/env python
"""
UNet 架构对比实验训练脚本（支持 WandB）

用法:
    # 训练单个架构
    python scripts/train_experiment.py --arch concatenation --epochs 100

    # 训练所有架构
    python scripts/train_experiment.py --arch all --epochs 100

    # 快速测试（少量epochs）
    python scripts/train_experiment.py --arch concatenation --epochs 5 --quick-test

    # 禁用 wandb
    python scripts/train_experiment.py --arch concatenation --no-wandb
"""
import os
import sys
import gc
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    HAS_EMA = False
    print("⚠️  torch-ema not installed, EMA disabled")


def parse_args():
    parser = argparse.ArgumentParser(description="UNet架构对比实验")

    parser.add_argument('--arch', type=str, default='concatenation',
                        choices=['concatenation', 'skip', 'cross_attention', 'all'],
                        help='UNet架构类型，或 "all" 训练全部')

    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--base-channels', type=int, default=256,
                        help='UNet基础通道数')

    parser.add_argument('--vae-path', type=str,
                        default='checkpoints/exp_concatenation_noCFG/vae.pth',
                        help='VAE模型路径')
    parser.add_argument('--manifest', type=str,
                        default='data/dataset/manifest.csv',
                        help='数据manifest路径')

    parser.add_argument('--early-stopping', type=int, default=None,
                        help='早停轮数（None表示禁用）')
    parser.add_argument('--quick-test', action='store_true',
                        help='快速测试模式（减少数据量）')

    # Resume 参数
    parser.add_argument('--resume', action='store_true',
                        help='从checkpoint恢复训练（需与时间戳目录配合使用）')

    # WandB 参数
    parser.add_argument('--no-wandb', action='store_true',
                        help='禁用 wandb')
    parser.add_argument('--wandb-project', type=str, default='cbct-sct-ldm',
                        help='WandB 项目名')

    # 混合精度参数
    parser.add_argument('--precision', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='训练精度: fp32 (慢但稳定), fp16 (快但可能NaN), bf16 (快且稳定, 推荐)')

    # 学习率调度策略
    parser.add_argument('--lr-schedule', type=str, default='wsd',
                        choices=['wsd', 'cosine'],
                        help='学习率调度: wsd (warmup-stable-decay, 稳定), cosine (warmup-cosine, 平滑)')

    # EMA 参数
    parser.add_argument('--use-ema', action='store_true',
                        help='启用 EMA (Exponential Moving Average)')
    parser.add_argument('--ema-decay', type=float, default=0.9995,
                        help='EMA衰减率 (推荐: 0.9995 适合中等训练长度, 0.9999 适合长训练)')

    # 训练优化参数
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument('--scheduler-patience', type=int, default=10,
                        help='学习率调度器patience (仅用于ReduceLROnPlateau)')

    return parser.parse_args()


def train_with_wandb(
    unet,
    vae,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    save_path,
    predict_dir,
    early_stopping,
    learning_rate,
    log_file,
    wandb_logger=None,
    arch_name="unet",
    precision="fp16",  # 'fp32', 'fp16', 'bf16'
    lr_schedule="wsd",  # 'wsd' or 'cosine'
    use_ema=True,
    ema_decay=0.9995,
    patience=10,
    gradient_clip_val=1.0,
    resume_from=None,  # checkpoint路径，用于恢复训练
):
    """
    带 WandB 集成的训练函数

    Args:
        wandb_logger: WandbLogger 实例（可为 None）
        arch_name: 架构名称（用于日志）
    """
    from models.diffusion import Diffusion
    from skimage.metrics import structural_similarity as ssim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    # 学习率调度策略
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    if lr_schedule == 'wsd':
        # Warmup-Stable-Decay
        stable_steps = int(0.70 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            elif current_step < stable_steps:
                return 1.0
            else:
                decay_steps = total_steps - stable_steps
                progress = (current_step - stable_steps) / max(1, decay_steps)
                return max(0.0, 1.0 - progress)

        print(f"✓ WSD Scheduler: warmup={warmup_steps}, stable={stable_steps}, total={total_steps} steps")

    else:  # cosine
        # Warmup + Cosine Annealing
        import math

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        print(f"✓ Cosine Scheduler: warmup={warmup_steps}, total={total_steps} steps")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    diffusion = Diffusion(device)

    # Mixed Precision
    use_amp = precision != 'fp32'
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
    scaler = GradScaler() if precision == 'fp16' else None

    if use_amp:
        print(f"✓ Mixed Precision ({precision.upper()}) enabled")
    else:
        print("✓ Full Precision (FP32)")

    # EMA
    ema = None
    if use_ema and HAS_EMA:
        ema = ExponentialMovingAverage(unet.parameters(), decay=ema_decay)
        print(f"✓ EMA enabled (decay={ema_decay})")

    best_val_loss = float('inf')
    early_stopping_counter = 0
    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = save_path.replace('.pth', '_checkpoint.pth')
    if resume_from and os.path.exists(resume_from):
        print(f"📂 恢复训练: {resume_from}")
        # 先加载到CPU，避免同时在GPU上持有checkpoint和模型
        checkpoint = torch.load(resume_from, map_location='cpu')
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])

        # 立即释放checkpoint，减少内存占用
        torch.cuda.empty_cache()

        print(f"✓ 从 epoch {start_epoch} 恢复，best_val_loss={best_val_loss:.4f}")

    # 记录损失历史（用于 wandb 表格）
    # 🔥 修复: 只在启用wandb时记录,避免不必要的内存占用
    train_losses = [] if wandb_logger else None
    val_losses = [] if wandb_logger else None

    for epoch in range(start_epoch, epochs):
        unet.train()
        train_loss = 0
        num_batches = 0

        # 训练循环
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for ct, cbct in progress_bar:
            ct = ct.to(device)
            cbct = cbct.to(device)

            with torch.no_grad():
                ct_z_mu, ct_z_logvar = vae.encode(ct)
                ct_z = vae.reparameterize(ct_z_mu, ct_z_logvar)
                cbct_z_mu, cbct_z_logvar = vae.encode(cbct)
                cbct_z = vae.reparameterize(cbct_z_mu, cbct_z_logvar)

            # 🔍 第一个batch打印latent范围，用于验证VAE匹配
            if epoch == start_epoch and num_batches == 0:
                print("\n" + "="*70)
                print("📊 VAE Latent空间验证 (第一个batch)")
                print("="*70)
                print(f"CT latent   - Range: [{ct_z.min():.2f}, {ct_z.max():.2f}], Mean: {ct_z.mean():.2f}, Std: {ct_z.std():.2f}")
                print(f"CBCT latent - Range: [{cbct_z.min():.2f}, {cbct_z.max():.2f}], Mean: {cbct_z.mean():.2f}, Std: {cbct_z.std():.2f}")
                print("✅ 预期: 均值≈0, 标准差≈1, 范围≈[-3, 3] (正态分布3σ)")
                print("="*70 + "\n")

            optimizer.zero_grad()

            # Forward pass (with AMP if enabled)
            if use_amp:
                with autocast(dtype=dtype):  # PyTorch 2.0兼容
                    t = diffusion.sample_timesteps(ct_z.size(0))
                    noise = torch.randn_like(ct_z)
                    ct_z_noisy = diffusion.add_noise(ct_z, t, noise=noise)
                    pred_noise = unet(ct_z_noisy, cbct_z, t)
                    loss = F.mse_loss(pred_noise, noise)
            else:
                t = diffusion.sample_timesteps(ct_z.size(0))
                noise = torch.randn_like(ct_z)
                ct_z_noisy = diffusion.add_noise(ct_z, t, noise=noise)
                pred_noise = unet(ct_z_noisy, cbct_z, t)
                loss = F.mse_loss(pred_noise, noise)

            # 🔥 修复: NaN检测 - 直接跳过，不更新任何状态
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf detected at epoch {epoch+1}, batch {num_batches}, skipping batch")
                continue  # 直接跳过这个batch，不做任何状态更新

            # Backward pass
            if scaler is not None:  # FP16
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:  # BF16/FP32
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=gradient_clip_val)
                optimizer.step()

            if ema is not None:
                ema.update()

            scheduler.step()

            loss_value = loss.item()
            train_loss += loss_value
            num_batches += 1

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': loss_value, 'lr': f'{current_lr:.2e}'})

        train_loss /= max(num_batches, 1)

        # 验证
        unet.eval()
        val_loss = 0
        val_batches = 0
        val_generator = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            for ct, cbct in val_loader:
                ct = ct.to(device)
                cbct = cbct.to(device)

                ct_z_mu, ct_z_logvar = vae.encode(ct)
                ct_z = vae.reparameterize(ct_z_mu, ct_z_logvar)
                cbct_z_mu, cbct_z_logvar = vae.encode(cbct)
                cbct_z = vae.reparameterize(cbct_z_mu, cbct_z_logvar)

                t = diffusion.sample_timesteps(ct_z.size(0), generator=val_generator)
                noise = torch.randn(ct_z.size(), dtype=ct_z.dtype, device=ct_z.device, generator=val_generator)
                ct_z_noisy = diffusion.add_noise(ct_z, t, noise)
                pred_noise = unet(ct_z_noisy, cbct_z, t)
                loss = F.mse_loss(pred_noise, noise)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录损失
        # 🔥 修复: 只在启用wandb时记录
        if wandb_logger:
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # WandB 日志
        if wandb_logger:
            wandb_logger.log_training_step(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                extra_metrics={
                    "best_val_loss": best_val_loss,
                    "early_stopping_counter": early_stopping_counter,
                }
            )

        # 打印日志
        log_message = (
            f"Epoch {epoch+1} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Gap: {val_loss - train_loss:+.4f} | "
            f"LR: {current_lr:.2e}"
        )
        print(log_message)

        if log_file:
            with open(log_file, 'a') as f:
                f.write(log_message + "\n")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0

            # 将模型参数转移到CPU以减少GPU显存占用
            if ema is not None:
                with ema.average_parameters():
                    state_dict_cpu = {k: v.cpu() for k, v in unet.state_dict().items()}
            else:
                state_dict_cpu = {k: v.cpu() for k, v in unet.state_dict().items()}

            # 保存带epoch标签的最佳模型
            best_model_path = save_path.replace('.pth', f'_ep{epoch+1:03d}.pth')
            torch.save(state_dict_cpu, best_model_path)

            # 同时更新unet_best.pth (覆盖旧的最佳模型)
            unet_best_path = save_path.replace('.pth', '_best.pth')
            torch.save(state_dict_cpu, unet_best_path)

            save_message = f"✅ Best model at epoch {epoch+1}, val_loss={val_loss:.4f} (saved: {os.path.basename(best_model_path)} & unet_best.pth)"
            print(save_message)

            if log_file:
                with open(log_file, 'a') as f:
                    f.write(save_message + "\n")

            # 保存到 wandb
            if wandb_logger:
                wandb_logger.log_metrics({
                    "best_epoch": epoch + 1,
                    "best_val_loss": val_loss,
                }, step=epoch + 1)
        else:
            early_stopping_counter += 1

        # 早停检查
        if early_stopping and early_stopping_counter >= early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # 定期生成预测样本
        if predict_dir and (epoch + 1) % 20 == 0:
            generate_predictions(
                unet, vae, test_loader, diffusion, device,
                epoch + 1, predict_dir, wandb_logger, ema
            )

        # 每10个epoch保存checkpoint（用于恢复训练）
        if (epoch + 1) % 10 == 0:
            try:
                # 将所有state_dict转移到CPU
                model_state_cpu = {k: v.cpu() for k, v in unet.state_dict().items()}

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_cpu,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'early_stopping_counter': early_stopping_counter,
                }
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                if ema is not None:
                    # EMA state_dict包含tensors和其他类型（如float），只转移tensors到CPU
                    ema_state = ema.state_dict()
                    checkpoint['ema_state_dict'] = {
                        k: v.cpu() if torch.is_tensor(v) else v
                        for k, v in ema_state.items()
                    }

                # 保存带epoch标签的checkpoint (只保留最新的)
                checkpoint_with_epoch = checkpoint_path.replace('_checkpoint.pth', f'_last_ep{epoch+1:03d}.pth')

                # 删除旧的last checkpoint
                checkpoint_dir = os.path.dirname(checkpoint_path)
                for old_ckpt in os.listdir(checkpoint_dir):
                    if old_ckpt.startswith('unet_last_ep') and old_ckpt.endswith('.pth'):
                        os.remove(os.path.join(checkpoint_dir, old_ckpt))

                torch.save(checkpoint, checkpoint_with_epoch)
                print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_with_epoch)}")

            except Exception as e:
                print(f"⚠️  Checkpoint保存失败: {e}")
                print(f"   训练继续，best model已保存")

    # 训练结束，记录最终结果
    if wandb_logger:
        wandb_logger.create_loss_table(train_losses, val_losses)

    return best_val_loss


def generate_predictions(unet, vae, test_loader, diffusion, device, epoch, predict_dir, wandb_logger, ema, num_samples=4):
    """
    生成预测样本并上传到 wandb

    Args:
        num_samples: 随机采样的样本数量
    """
    import torchvision
    import random

    unet.eval()
    os.makedirs(os.path.join(predict_dir, f"epoch_{epoch}"), exist_ok=True)

    # 随机选择样本索引
    dataset = test_loader.dataset
    total_samples = len(dataset)
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    for sample_idx, data_idx in enumerate(random_indices):
        ct, cbct = dataset[data_idx]
        ct = ct.unsqueeze(0).to(device)
        cbct = cbct.unsqueeze(0).to(device)

        with torch.no_grad():
            # 编码条件
            cbct_z_mu, cbct_z_logvar = vae.encode(cbct)
            cbct_z = vae.reparameterize(cbct_z_mu, cbct_z_logvar)

            # DDIM 采样
            ddim_steps = 40
            timesteps = list(np.linspace(0, 999, ddim_steps, dtype=int)[::-1])

            z = torch.randn_like(cbct_z)
            alpha_cumprod = diffusion.alpha_cumprod.to(device)

            # 🔥 修复: 使用try-finally确保EMA restore,防止显存泄漏
            ema_backed_up = False
            try:
                if ema is not None:
                    ema.store()
                    ema.copy_to()
                    ema_backed_up = True

                for i in range(len(timesteps) - 1):
                    t = timesteps[i]
                    t_prev = timesteps[i + 1]
                    t_tensor = torch.full((z.size(0),), t, device=device, dtype=torch.long)

                    eps = unet(z, cbct_z, t_tensor)

                    a_t = alpha_cumprod[t]
                    a_prev = alpha_cumprod[t_prev]
                    x0_pred = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
                    z = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * eps
            finally:
                # 🔥 确保EMA restore被调用
                if ema_backed_up:
                    ema.restore()

            # 解码
            sct = vae.decode(z)

            # 转换为图像格式
            ct_img = (ct[0] / 2 + 0.5).clamp(0, 1)
            cbct_img = (cbct[0] / 2 + 0.5).clamp(0, 1)
            sct_img = (sct[0] / 2 + 0.5).clamp(0, 1)

            # 保存到本地
            images_to_save = [cbct_img, ct_img, sct_img]
            output_path = os.path.join(
                predict_dir, f"epoch_{epoch}",
                f"sample_{sample_idx}_idx_{data_idx}.png"
            )
            torchvision.utils.save_image(images_to_save, output_path, nrow=3)

            # 上传到 wandb（每个样本都上传）
            if wandb_logger:
                # 🔥 修复: 立即转CPU并复制,避免持有GPU tensor引用
                ct_np = ct_img.detach().cpu().numpy().copy().squeeze()
                cbct_np = cbct_img.detach().cpu().numpy().copy().squeeze()
                sct_np = sct_img.detach().cpu().numpy().copy().squeeze()

                wandb_logger.log_comparison_images(
                    cbct=cbct_np,
                    ct_gt=ct_np,
                    sct_pred=sct_np,
                    step=epoch,
                    sample_idx=sample_idx
                )

                # 🔥 修复: 立即删除numpy数组,释放CPU内存
                del ct_np, cbct_np, sct_np

    # 清理GPU缓存
    torch.cuda.empty_cache()
    print(f"✅ Predictions saved to {predict_dir}/epoch_{epoch} ({num_samples} random samples)")


def train_architecture(arch_name, args, wandb_logger=None):
    """训练指定架构"""
    from utils.dataset_256 import get_dataloaders_256, PairedCTCBCTDatasetNPY256
    from models.vae import load_vae
    from models.unetConditional import UNetSkip, UNetConcatenation, UNetCrossAttention

    print("\n" + "=" * 70)
    print(f"训练架构: {arch_name}")
    print("=" * 70)

    # 生成带时间戳的实验目录名 (格式: arch_noCFG_timestamp_ep200)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir_name = f"{arch_name}_noCFG_{timestamp}_ep{args.epochs}"
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", exp_dir_name)

    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)

    print(f"📁 实验目录: {exp_dir_name}")

    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders_256(
        manifest_path=os.path.join(PROJECT_ROOT, args.manifest),
        batch_size=args.batch_size,
        num_workers=4,
        dataset_class=PairedCTCBCTDatasetNPY256,
        target_size=(256, 256),
        train_size=500 if args.quick_test else None,
        val_size=100 if args.quick_test else None,
        test_size=100 if args.quick_test else None,
        augmentation={
            'degrees': (-3, 3),         # ±3° 旋转
            'translate': (0.1, 0.1),    # 10% 平移
            'scale': (0.9, 1.1),        # 10% 缩放
        },
        preprocess="linear"
    )
    print(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")

    # 加载 VAE
    print(f"\n加载 VAE: {args.vae_path}")
    vae = load_vae(args.vae_path, trainable=False)

    # 选择架构
    arch_map = {
        'concatenation': UNetConcatenation,
        'skip': UNetSkip,
        'cross_attention': UNetCrossAttention,
    }

    UNetClass = arch_map[arch_name]
    print(f"使用架构: {UNetClass.__name__}")

    # 初始化 UNet
    unet = UNetClass(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        dropout_rate=0.1,
    ).cuda()

    param_count = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"参数量: {param_count:.2f}M")

    # 保存路径
    unet_save_path = os.path.join(save_dir, "unet.pth")
    checkpoint_path = os.path.join(save_dir, "unet_checkpoint.pth")
    log_file = os.path.join(save_dir, "logs", "unet_training.log")
    predict_dir = os.path.join(save_dir, "predictions")

    # 检查是否需要恢复训练
    resume_from = None
    if args.resume and os.path.exists(checkpoint_path):
        resume_from = checkpoint_path
        print(f"📂 找到checkpoint: {checkpoint_path}")

    # 写入日志头（append模式如果是resume）
    log_mode = 'a' if resume_from else 'w'
    with open(log_file, log_mode) as f:
        if resume_from:
            f.write(f"\n# 恢复训练: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            f.write(f"# UNet架构对比实验 - {arch_name}\n")
            f.write(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 参数: epochs={args.epochs}, batch_size={args.batch_size}, ")
            f.write(f"base_channels={args.base_channels}, lr={args.lr}\n")
            f.write(f"# 架构: {UNetClass.__name__}, 参数量: {param_count:.2f}M\n\n")

    # 配置 wandb
    if wandb_logger:
        wandb_logger.log_config({
            "architecture": arch_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "base_channels": args.base_channels,
            "param_count_M": param_count,
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "precision": args.precision,
            "lr_schedule": args.lr_schedule,
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay,
            "gradient_clip": args.gradient_clip,
        })

    # 训练
    print("\n开始训练...")
    best_val_loss = train_with_wandb(
        unet=unet,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        save_path=unet_save_path,
        predict_dir=predict_dir,
        early_stopping=args.early_stopping,
        learning_rate=args.lr,
        log_file=log_file,
        wandb_logger=wandb_logger,
        arch_name=arch_name,
        precision=args.precision,
        lr_schedule=args.lr_schedule,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        gradient_clip_val=args.gradient_clip,
        patience=args.scheduler_patience,
        resume_from=resume_from,
    )

    print(f"\n✅ 架构 {arch_name} 训练完成")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    print(f"   模型保存: {unet_save_path}")
    print(f"   日志: {log_file}")

    # 训练完成后释放GPU内存，为下一个架构腾出空间
    del unet, vae, train_loader, val_loader, test_loader
    if wandb_logger:
        wandb_logger.finish()
        del wandb_logger

    # 强制清理GPU缓存和Python对象
    gc.collect()
    torch.cuda.empty_cache()

    print("🧹 GPU内存已释放")

    return save_dir, best_val_loss


def main():
    args = parse_args()

    print("=" * 70)
    print("UNet 架构对比实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 检查 VAE
    if not os.path.exists(os.path.join(PROJECT_ROOT, args.vae_path)):
        print(f"❌ 错误: 找不到 VAE 模型: {args.vae_path}")
        return

    # 确定要训练的架构
    if args.arch == 'all':
        architectures = ['concatenation', 'skip', 'cross_attention']
    else:
        architectures = [args.arch]

    print(f"\n将训练以下架构: {architectures}")
    print(f"每个架构训练 {args.epochs} epochs")
    print(f"WandB: {'禁用' if args.no_wandb else '启用'}")
    print(f"Resume: {'启用' if args.resume else '禁用'}")

    if args.quick_test:
        print("⚡ 快速测试模式")

    # 训练每个架构
    results = {}
    for arch in architectures:
        # 初始化 wandb
        wandb_logger = None
        if not args.no_wandb:
            from utils.wandb_logger import WandbLogger
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                name=f"arch_{arch}",
                config={
                    "experiment_type": "architecture_comparison",
                    "architecture": arch,
                    "cfg": False,
                },
                tags=[arch, "no-cfg", "architecture-comparison"],
            )

        try:
            save_dir, best_loss = train_architecture(arch, args, wandb_logger)
            results[arch] = (save_dir, best_loss)
        except Exception as e:
            print(f"\n❌ 架构 {arch} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[arch] = (None, None)
        finally:
            if wandb_logger:
                wandb_logger.finish()

    # 总结
    print("\n" + "=" * 70)
    print("实验完成总结")
    print("=" * 70)
    for arch, (path, loss) in results.items():
        if path:
            loss_str = f", best_val_loss={loss:.4f}" if loss else ""
            print(f"✅ {arch}: {path}{loss_str}")
        else:
            print(f"❌ {arch}: 失败")

    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n下一步: 运行评估脚本比较各架构效果")
    print("python scripts/eval_experiment.py --experiments all")


if __name__ == '__main__':
    main()
