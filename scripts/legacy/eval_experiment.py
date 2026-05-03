#!/usr/bin/env python
"""
对比实验评估脚本

用法:
    # 评估所有架构
    python scripts/eval_experiment.py --experiments all

    # 评估特定架构
    python scripts/eval_experiment.py --experiments concatenation skip

    # 指定采样步数
    python scripts/eval_experiment.py --experiments all --ddim-steps 40

    # 生成对比报告
    python scripts/eval_experiment.py --experiments all --report
"""
import os
import sys
import argparse
import json
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="对比实验评估")

    parser.add_argument('--experiments', type=str, nargs='+', default=['all'],
                        help='要评估的实验，或 "all"')

    parser.add_argument('--ddim-steps', type=int, default=40,
                        help='DDIM采样步数')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='评估样本数')
    parser.add_argument('--cfg-scale', type=float, default=1.0,
                        help='CFG scale (1.0=无CFG)')

    parser.add_argument('--report', action='store_true',
                        help='生成对比报告')
    parser.add_argument('--output-dir', type=str, default='outputs/experiments',
                        help='输出目录')

    return parser.parse_args()


def compute_metrics(pred, gt):
    """计算评估指标"""
    from skimage.metrics import structural_similarity as ssim

    mae = np.mean(np.abs(pred - gt))
    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    psnr = 10 * np.log10(2500**2 / mse) if mse > 0 else float('inf')
    ssim_val = ssim(pred, gt, data_range=2500)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'PSNR': psnr,
        'SSIM': ssim_val,
    }


def evaluate_experiment(exp_name, args, arch_dir=None):
    """评估单个实验

    Args:
        exp_name: 架构名称 (skip, concatenation, cross_attention)
        args: 命令行参数
        arch_dir: 可选的具体目录名，如果为None则自动查找
    """
    from models.vae import load_vae
    from models.unetConditional import UNetSkip, UNetConcatenation, UNetCrossAttention
    from models.diffusion import Diffusion
    from utils.dataset_256 import get_dataloaders_256, PairedCTCBCTDatasetNPY256

    print(f"\n{'='*60}")
    print(f"评估: {exp_name}")
    print(f"{'='*60}")

    # 自动查找模型目录
    if arch_dir is None:
        # 尝试不同的目录命名模式
        checkpoint_base = os.path.join(PROJECT_ROOT, "checkpoints")
        possible_dirs = [
            f"exp_{exp_name}_noCFG",  # 标准模式
        ]
        # 查找包含架构名和noCFG的目录
        for d in os.listdir(checkpoint_base):
            if exp_name in d and 'noCFG' in d:
                possible_dirs.insert(0, d)  # 优先使用找到的目录

        exp_dir = None
        for d in possible_dirs:
            test_dir = os.path.join(checkpoint_base, d)
            if os.path.exists(test_dir):
                exp_dir = test_dir
                break
    else:
        exp_dir = os.path.join(PROJECT_ROOT, "checkpoints", arch_dir)

    if exp_dir is None or not os.path.exists(exp_dir):
        print(f"❌ 未找到目录: {exp_name}")
        return None

    print(f"使用目录: {exp_dir}")

    vae_path = os.path.join(exp_dir, "vae.pth")
    unet_path = os.path.join(exp_dir, "unet.pth")

    # VAE 可能在共享目录
    if not os.path.exists(vae_path):
        # 尝试其他可能的VAE位置
        vae_alternatives = [
            os.path.join(PROJECT_ROOT, "checkpoints/trained_models_256/vae.pth"),
            os.path.join(PROJECT_ROOT, "checkpoints/exp_concatenation_noCFG/vae.pth"),
        ]
        for alt_path in vae_alternatives:
            if os.path.exists(alt_path):
                vae_path = alt_path
                break

    if not os.path.exists(unet_path):
        print(f"❌ 未找到UNet模型: {unet_path}")
        return None

    # 加载模型
    print(f"加载 VAE: {vae_path}")
    vae = load_vae(vae_path, trainable=False).cuda().eval()

    print(f"加载 UNet: {unet_path}")

    # 根据实验名选择架构
    arch_map = {
        'concatenation': UNetConcatenation,
        'skip': UNetSkip,
        'cross_attention': UNetCrossAttention,
    }

    UNetClass = arch_map.get(exp_name)
    if UNetClass is None:
        print(f"❌ 未知架构: {exp_name}")
        return None

    # 根据架构确定base_channels（Cross Attention使用192，其他使用256）
    if exp_name == 'cross_attention':
        base_ch = 192
    else:
        base_ch = 256

    unet = UNetClass(
        in_channels=3, out_channels=3,
        base_channels=base_ch, dropout_rate=0.1
    ).cuda()
    unet.load_state_dict(torch.load(unet_path, map_location='cuda'))
    unet.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = Diffusion(device=device, timesteps=1000)

    # 加载测试数据
    _, _, test_loader = get_dataloaders_256(
        manifest_path=os.path.join(PROJECT_ROOT, "data/dataset/manifest.csv"),
        batch_size=1,
        num_workers=2,
        dataset_class=PairedCTCBCTDatasetNPY256,
        target_size=(256, 256),
        test_size=args.num_samples,
        preprocess="linear"
    )

    # DDIM 采样
    timesteps = list(np.linspace(0, 999, args.ddim_steps, dtype=int)[::-1])

    all_metrics = []
    print(f"\n推理 {len(test_loader)} 个样本 (DDIM {args.ddim_steps}步)...")

    for batch in tqdm(test_loader, desc="评估中"):
        ct, cbct = batch
        ct = ct.cuda()
        cbct = cbct.cuda()

        with torch.no_grad():
            # 编码
            ct_latent, _, _, _ = vae(ct)
            cbct_latent, _, _, _ = vae(cbct)

            # DDIM 采样
            x = torch.randn_like(ct_latent)
            alphas = diffusion.alpha
            alpha_bars = diffusion.alpha_cumprod

            for i, t in enumerate(timesteps):
                t_tensor = torch.tensor([t], device='cuda').long()
                noise_pred = unet(x, cbct_latent, t_tensor)

                alpha_bar_t = alpha_bars[t]
                if i + 1 < len(timesteps):
                    alpha_bar_t_prev = alpha_bars[timesteps[i + 1]]
                else:
                    alpha_bar_t_prev = torch.tensor(1.0, device='cuda')

                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                x0_pred = torch.clamp(x0_pred, -1, 1)

                if i + 1 < len(timesteps):
                    dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
                    x = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt
                else:
                    x = x0_pred

            # 解码
            sct = vae.decode(x)

        # 转换回 HU
        ct_hu = ct.cpu().numpy().squeeze() * 1250 + 250
        sct_hu = sct.cpu().numpy().squeeze() * 1250 + 250
        sct_hu = np.clip(sct_hu, -1000, 1500)

        # 计算指标
        metrics = compute_metrics(sct_hu, ct_hu)
        all_metrics.append(metrics)

    # 汇总
    summary = {}
    for key in ['MAE', 'RMSE', 'PSNR', 'SSIM']:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }

    print(f"\n结果:")
    print(f"  MAE:  {summary['MAE']['mean']:.2f} ± {summary['MAE']['std']:.2f} HU")
    print(f"  PSNR: {summary['PSNR']['mean']:.2f} ± {summary['PSNR']['std']:.2f} dB")
    print(f"  SSIM: {summary['SSIM']['mean']:.4f} ± {summary['SSIM']['std']:.4f}")

    return summary


def generate_report(results, args):
    """生成对比报告"""
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'ddim_steps': args.ddim_steps,
            'num_samples': args.num_samples,
            'cfg_scale': args.cfg_scale,
        },
        'results': results,
    }

    json_path = os.path.join(output_dir, 'architecture_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ JSON 报告: {json_path}")

    # 打印对比表格
    print("\n" + "=" * 70)
    print("架构对比结果")
    print("=" * 70)
    print(f"{'架构':<20} {'MAE (HU)':<15} {'PSNR (dB)':<15} {'SSIM':<15}")
    print("-" * 70)

    for arch, metrics in sorted(results.items(), key=lambda x: x[1]['MAE']['mean'] if x[1] else float('inf')):
        if metrics is None:
            print(f"{arch:<20} {'失败':<15}")
        else:
            mae = f"{metrics['MAE']['mean']:.1f}±{metrics['MAE']['std']:.1f}"
            psnr = f"{metrics['PSNR']['mean']:.1f}±{metrics['PSNR']['std']:.1f}"
            ssim_v = f"{metrics['SSIM']['mean']:.4f}±{metrics['SSIM']['std']:.4f}"
            print(f"{arch:<20} {mae:<15} {psnr:<15} {ssim_v:<15}")

    print("=" * 70)

    # 找出最佳
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_mae = min(valid_results.items(), key=lambda x: x[1]['MAE']['mean'])
        best_psnr = max(valid_results.items(), key=lambda x: x[1]['PSNR']['mean'])
        best_ssim = max(valid_results.items(), key=lambda x: x[1]['SSIM']['mean'])

        print(f"\n最佳 MAE:  {best_mae[0]} ({best_mae[1]['MAE']['mean']:.1f} HU)")
        print(f"最佳 PSNR: {best_psnr[0]} ({best_psnr[1]['PSNR']['mean']:.1f} dB)")
        print(f"最佳 SSIM: {best_ssim[0]} ({best_ssim[1]['SSIM']['mean']:.4f})")


def main():
    args = parse_args()

    print("=" * 70)
    print("对比实验评估")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 确定要评估的实验
    if 'all' in args.experiments:
        # 自动检测已有实验（支持多种命名模式）
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        experiments = []
        arch_dirs = {}  # 架构名 -> 目录名映射

        for d in os.listdir(checkpoint_dir):
            # 模式1: exp_<arch>_noCFG
            if d.startswith('exp_') and 'noCFG' in d:
                arch = d.replace('exp_', '').split('_noCFG')[0]
                experiments.append(arch)
                arch_dirs[arch] = d
            # 模式2: <arch>_noCFG_<timestamp>_ep<num>
            elif 'noCFG' in d and not d.startswith('exp_'):
                # 例如: skip_noCFG_20260120_154721_ep200
                arch = d.split('_noCFG')[0]
                if arch in ['skip', 'concatenation', 'cross_attention']:
                    experiments.append(arch)
                    arch_dirs[arch] = d

        # 去重
        experiments = list(set(experiments))
    else:
        experiments = args.experiments
        arch_dirs = None

    print(f"\n将评估: {experiments}")
    print(f"DDIM 步数: {args.ddim_steps}")
    print(f"样本数: {args.num_samples}")

    # 评估每个实验
    results = {}
    for exp in experiments:
        try:
            # 如果有arch_dirs映射，传递具体目录名
            if arch_dirs and exp in arch_dirs:
                results[exp] = evaluate_experiment(exp, args, arch_dir=arch_dirs[exp])
            else:
                results[exp] = evaluate_experiment(exp, args)
        except Exception as e:
            import traceback
            print(f"❌ 评估 {exp} 失败: {e}")
            print("\n完整错误信息:")
            traceback.print_exc()
            results[exp] = None

    # 生成报告
    if args.report or len(results) > 1:
        generate_report(results, args)


if __name__ == '__main__':
    main()
