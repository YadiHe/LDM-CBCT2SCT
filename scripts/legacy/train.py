#!/usr/bin/env python
"""
统一训练入口脚本

用法:
    # 训练 VAE + UNet (默认配置)
    python scripts/train.py

    # 只训练 UNet (使用已有VAE)
    python scripts/train.py --stage unet --vae-path checkpoints/trained_models_256/vae.pth

    # 训练 CFG 版本
    python scripts/train.py --stage unet --use-cfg --save-dir trained_models_cfg

    # 指定 UNet 类型
    python scripts/train.py --unet-type skip

    # 断点续训
    python scripts/train.py --resume checkpoints/trained_models_256/unet.pth
"""
import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from configs.base import DataConfig, ModelConfig, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="统一训练脚本")

    # 训练阶段
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'vae', 'unet'],
                        help='训练阶段: all=VAE+UNet, vae=仅VAE, unet=仅UNet')

    # 数据配置
    parser.add_argument('--manifest', type=str, default='dataset/manifest.csv',
                        help='数据manifest文件路径')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--target-size', type=int, default=256,
                        help='目标图像尺寸 (256 或 512)')

    # 模型配置
    parser.add_argument('--unet-type', type=str, default='concatenation',
                        choices=['concatenation', 'skip', 'cross_attention', 'control_paca'],
                        help='UNet条件融合类型')
    parser.add_argument('--base-channels', type=int, default=256,
                        help='UNet基础通道数')
    parser.add_argument('--use-cfg', action='store_true',
                        help='启用Classifier-Free Guidance训练')
    parser.add_argument('--cfg-dropout', type=float, default=0.15,
                        help='CFG条件丢弃率')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--early-stopping', type=int, default=30,
                        help='早停轮数')
    parser.add_argument('--use-fp16', action='store_true', default=True,
                        help='使用混合精度训练')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='使用EMA')

    # 路径配置
    parser.add_argument('--save-dir', type=str, default='trained_models_256',
                        help='模型保存目录')
    parser.add_argument('--vae-path', type=str, default=None,
                        help='已训练VAE路径 (用于只训练UNet)')
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续训checkpoint路径')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')

    return parser.parse_args()


def setup_configs(args):
    """根据命令行参数设置配置"""
    data_config = DataConfig(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.target_size, args.target_size),
    )

    model_config = ModelConfig(
        unet_base_channels=args.base_channels,
        unet_type=args.unet_type,
        use_cfg=args.use_cfg,
        cfg_dropout_rate=args.cfg_dropout,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping=args.early_stopping,
        use_fp16=args.use_fp16,
        use_ema=args.use_ema,
        save_dir=args.save_dir,
    )

    return data_config, model_config, training_config


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置配置
    data_config, model_config, training_config = setup_configs(args)

    # 创建保存目录
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", training_config.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)

    print("=" * 60)
    print(f"CBCT → sCT 训练脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\n训练阶段: {args.stage}")
    print(f"UNet类型: {model_config.unet_type}")
    print(f"CFG: {'启用' if model_config.use_cfg else '禁用'}")
    print(f"保存目录: {save_dir}")
    print("=" * 60)

    # 导入训练模块
    from utils.dataset_256 import get_dataloaders_256, PairedCTCBCTDatasetNPY256
    from models.vae import load_vae, train_vae

    # 加载数据
    print("\n加载数据集...")
    manifest_path = os.path.join(PROJECT_ROOT, data_config.manifest_path)
    train_loader, val_loader, test_loader = get_dataloaders_256(
        manifest_path=manifest_path,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        dataset_class=PairedCTCBCTDatasetNPY256,
        target_size=data_config.target_size,
        augmentation=data_config.augmentation,
        preprocess=data_config.preprocess,
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # ========== 阶段1: VAE训练 ==========
    if args.stage in ['all', 'vae']:
        print("\n" + "=" * 60)
        print("阶段 1: VAE 训练")
        print("=" * 60)

        vae_save_path = os.path.join(save_dir, "vae.pth")
        vae_log_file = os.path.join(save_dir, "logs", "vae_training.log")

        vae = load_vae(save_path=None, trainable=True)
        train_vae(
            vae=vae,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=200,  # VAE通常训练较少轮数
            early_stopping=training_config.early_stopping,
            save_path=vae_save_path,
            log_file=vae_log_file,
        )
        print(f"✓ VAE 保存到: {vae_save_path}")

    # ========== 阶段2: UNet训练 ==========
    if args.stage in ['all', 'unet']:
        print("\n" + "=" * 60)
        print("阶段 2: 条件UNet 训练")
        print("=" * 60)

        # 确定VAE路径
        if args.vae_path:
            vae_path = args.vae_path
        else:
            vae_path = os.path.join(save_dir, "vae.pth")

        if not os.path.exists(vae_path):
            print(f"❌ 错误: 找不到VAE模型: {vae_path}")
            print("请先训练VAE或指定 --vae-path")
            return

        # 加载VAE（冻结）
        print(f"加载VAE: {vae_path}")
        vae = load_vae(save_path=vae_path, trainable=False)

        # 根据UNet类型选择模型
        unet_save_path = os.path.join(save_dir, "unet.pth")
        unet_log_file = os.path.join(save_dir, "logs", "unet_training.log")
        predict_dir = os.path.join(save_dir, "predictions")

        if model_config.use_cfg:
            # CFG版本
            from models.unetConditionalGuidance_ import load_cond_unet, train_cond_unet, UNetConcatenation
            print("使用 CFG 训练模式 (条件丢弃率: {})".format(model_config.cfg_dropout_rate))

            unet = load_cond_unet(
                save_path=args.resume,
                trainable=True,
                base_channels=model_config.unet_base_channels,
                unet_type=UNetConcatenation,
            )

            train_cond_unet(
                unet=unet,
                vae=vae,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=training_config.epochs,
                early_stopping=training_config.early_stopping,
                save_path=unet_save_path,
                predict_dir=predict_dir,
                learning_rate=training_config.learning_rate,
                log_file=unet_log_file,
                use_fp16=training_config.use_fp16,
                use_ema=training_config.use_ema,
            )
        else:
            # 标准版本
            from models.unetConditional import load_cond_unet, train_cond_unet
            print(f"使用标准训练模式 (UNet类型: {model_config.unet_type})")

            # 根据类型选择模型类
            if model_config.unet_type == "concatenation":
                from models.unetConditional import UNetConcatenation as UNetClass
            elif model_config.unet_type == "skip":
                from models.unetConditional import UNetSkip as UNetClass
            elif model_config.unet_type == "cross_attention":
                from models.unetConditional import UNetCrossAttention as UNetClass
            else:
                from models.unetConditional import UNetConcatenation as UNetClass

            unet = load_cond_unet(
                save_path=args.resume,
                trainable=True,
                base_channels=model_config.unet_base_channels,
                unet_type=UNetClass,
            )

            train_cond_unet(
                unet=unet,
                vae=vae,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=training_config.epochs,
                early_stopping=training_config.early_stopping,
                save_path=unet_save_path,
                predict_dir=predict_dir,
                learning_rate=training_config.learning_rate,
                log_file=unet_log_file,
            )

        print(f"✓ UNet 保存到: {unet_save_path}")

    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型保存在: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
