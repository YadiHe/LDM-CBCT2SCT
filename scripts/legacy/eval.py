#!/usr/bin/env python
"""
完整评估脚本 - 匹配10月版本的full evaluation逻辑

功能:
  1. HU范围评估 (Full Range, Soft Tissue, Low Density, High Density)
  2. 可视化生成 (Bland-Altman, Boxplot, Error Heatmap, Histograms)
  3. Markdown报告生成
  4. CSV/JSON结果输出

用法:
    python scripts/eval.py --pred outputs/inference/arch_name --gt data/test_set
    python scripts/eval.py --pred outputs/inference/arch_name --gt data/test_set --output outputs/evaluation --full
"""
import os
import sys
import argparse
import json
import csv
import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(output_dir):
    """设置日志"""
    log_file = os.path.join(output_dir, 'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# 评估指标计算
# ============================================================================

def compute_mae(pred, gt, mask=None):
    """计算平均绝对误差"""
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    return np.mean(np.abs(pred - gt))


def compute_rmse(pred, gt, mask=None):
    """计算均方根误差"""
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    return np.sqrt(np.mean((pred - gt) ** 2))


def compute_me(pred, gt, mask=None):
    """计算平均误差（偏差）"""
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    return np.mean(pred - gt)


def compute_psnr(pred, gt, data_range=2000.0, mask=None):
    """计算峰值信噪比"""
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def compute_ssim(pred, gt, data_range=2000.0):
    """计算结构相似性 - 匹配10月版本逻辑"""
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(pred, gt, data_range=data_range)
    except Exception:
        # 10月版本：任何失败都返回NaN（包括ImportError和计算错误）
        return np.nan


def evaluate_sample(pred, gt, data_range=2000.0):
    """评估单个样本"""
    metrics = {
        'MAE': float(compute_mae(pred, gt)),
        'RMSE': float(compute_rmse(pred, gt)),
        'ME': float(compute_me(pred, gt)),
        'PSNR': float(compute_psnr(pred, gt, data_range)),
        'SSIM': float(compute_ssim(pred, gt, data_range)),
    }
    return metrics


def evaluate_hu_ranges(pred, gt, data_range=2000.0):
    """HU范围评估 - 完全匹配10月版本逻辑

    10月版本在mask后的1D数组上调用SSIM，如果失败返回NaN
    """
    hu_ranges = {
        'Full Range': (-1000, 1000),
        'Soft Tissue': (-150, 150),
        'Low Density': (-1000, -150),
        'High Density': (150, 1000),
    }

    results = {}
    for range_name, (low, high) in hu_ranges.items():
        mask = (gt >= low) & (gt <= high)
        if mask.sum() == 0:
            continue

        # 10月版本：对mask后的数据计算所有指标（包括SSIM）
        # SSIM在1D数组上会失败，但compute_ssim会返回NaN
        pred_masked = pred[mask]
        gt_masked = gt[mask]

        results[range_name] = {
            'MAE': float(compute_mae(pred_masked, gt_masked)),
            'RMSE': float(compute_rmse(pred_masked, gt_masked)),
            'ME': float(compute_me(pred_masked, gt_masked)),
            'PSNR': float(compute_psnr(pred_masked, gt_masked, data_range)),
            'SSIM': float(compute_ssim(pred_masked, gt_masked, data_range)),  # 可能返回NaN
            'pixel_count': int(mask.sum()),
        }

    return results


# ============================================================================
# 可视化生成 - 10月版本逻辑
# ============================================================================

def generate_bland_altman_plot(all_metrics, output_dir, logger):
    """生成Bland-Altman图"""
    try:
        pred_values = []
        gt_values = []

        for m in all_metrics:
            if 'pred_mean' in m and 'gt_mean' in m:
                pred_values.append(m['pred_mean'])
                gt_values.append(m['gt_mean'])

        if len(pred_values) == 0:
            logger.warning("无法生成Bland-Altman图: 缺少数据")
            return

        pred_values = np.array(pred_values)
        gt_values = np.array(gt_values)

        mean_values = (pred_values + gt_values) / 2
        diff_values = pred_values - gt_values

        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(mean_values, diff_values, alpha=0.5, s=20)
        ax.axhline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.2f}')
        ax.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}')
        ax.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}')

        ax.set_xlabel('Mean of sCT and CT (HU)', fontsize=12)
        ax.set_ylabel('Difference (sCT - CT) (HU)', fontsize=12)
        ax.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(output_dir, 'bland_altman_plot.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Bland-Altman图已保存: {output_path}")
    except Exception as e:
        logger.error(f"生成Bland-Altman图失败: {e}")


def generate_boxplot(summary_hu_ranges, output_dir, logger):
    """生成箱线图"""
    try:
        if not summary_hu_ranges:
            logger.warning("无法生成箱线图: 缺少HU范围数据")
            return

        metrics_to_plot = ['MAE', 'RMSE', 'PSNR', 'SSIM']
        range_names = list(summary_hu_ranges.keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            data = []
            labels = []

            for range_name in range_names:
                if metric in summary_hu_ranges[range_name]:
                    # 使用mean值（因为我们只有汇总统计）
                    data.append([summary_hu_ranges[range_name][metric]['mean']])
                    labels.append(range_name.replace(' ', '\n'))

            if data:
                ax.boxplot(data, labels=labels)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f'{metric} by HU Range', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'boxplot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 箱线图已保存: {output_path}")
    except Exception as e:
        logger.error(f"生成箱线图失败: {e}")


def generate_error_heatmap(all_metrics, output_dir, logger):
    """生成误差热力图（简化版）"""
    try:
        mae_values = [m['MAE'] for m in all_metrics]

        if len(mae_values) == 0:
            logger.warning("无法生成误差热力图: 缺少数据")
            return

        # 创建简单的误差分布热力图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 将MAE值重塑为矩阵形式（如果样本数是完全平方数）
        n = len(mae_values)
        side = int(np.ceil(np.sqrt(n)))
        padded = mae_values + [0] * (side * side - n)
        matrix = np.array(padded).reshape(side, side)

        im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
        ax.set_title('MAE Distribution Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='MAE (HU)')

        output_path = os.path.join(output_dir, 'error_heatmap.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 误差热力图已保存: {output_path}")
    except Exception as e:
        logger.error(f"生成误差热力图失败: {e}")


def generate_histograms(all_metrics, output_dir, logger):
    """生成指标分布直方图"""
    try:
        metrics_to_plot = ['MAE', 'RMSE', 'PSNR', 'SSIM']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [m[metric] for m in all_metrics if metric in m]

            if values:
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
                ax.axvline(np.median(values), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.3f}')
                ax.set_xlabel(metric, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'histograms.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ 直方图已保存: {output_path}")
    except Exception as e:
        logger.error(f"生成直方图失败: {e}")


# ============================================================================
# Markdown报告生成 - 10月版本格式
# ============================================================================

def generate_markdown_report(summary, summary_hu_ranges, config, output_dir, logger):
    """生成Markdown评估报告"""
    try:
        report_path = os.path.join(output_dir, 'evaluation_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            # 标题
            f.write("# CBCT-to-SCT 评估报告\n\n")
            f.write(f"**生成时间:** {datetime.now()}\n\n")
            f.write("---\n\n")

            # HU范围评估结果
            f.write("## HU范围评估\n\n")

            for range_name in ['Full Range', 'Soft Tissue', 'Low Density', 'High Density']:
                if range_name not in summary_hu_ranges:
                    continue

                range_data = summary_hu_ranges[range_name]
                hu_low, hu_high = {
                    'Full Range': (-1000, 1000),
                    'Soft Tissue': (-150, 150),
                    'Low Density': (-1000, -150),
                    'High Density': (150, 1000),
                }[range_name]

                f.write(f"\n### {range_name} [{hu_low}, {hu_high}] HU\n\n")
                f.write("| Metric | Mean | Std | Median |\n")
                f.write("|--------|------|-----|--------|\n")

                for metric in ['MAE', 'RMSE', 'ME', 'PSNR', 'SSIM']:
                    if metric in range_data:
                        mean_val = range_data[metric]['mean']
                        std_val = range_data[metric]['std']
                        median_val = range_data[metric]['median']
                        f.write(f"| {metric} | {mean_val:.3f} | {std_val:.3f} | {median_val:.3f} |\n")

                f.write("\n\n")

            # 配置信息
            f.write("---\n\n")
            f.write("## 评估配置\n\n")
            f.write(f"- **预测目录**: {config['pred_dir']}\n")
            f.write(f"- **GT目录**: {config['gt_dir']}\n")
            f.write(f"- **样本数**: {config['num_samples']}\n")
            f.write(f"- **数据范围**: {config['data_range']} HU\n\n")

            # 整体指标
            f.write("---\n\n")
            f.write("## 整体指标\n\n")
            f.write("| Metric | Mean | Std | Min | Max |\n")
            f.write("|--------|------|-----|-----|-----|\n")

            for metric in ['MAE', 'RMSE', 'ME', 'PSNR', 'SSIM']:
                if metric in summary:
                    s = summary[metric]
                    f.write(f"| {metric} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |\n")

            f.write("\n")

        logger.info(f"✓ Markdown报告已保存: {report_path}")
    except Exception as e:
        logger.error(f"生成Markdown报告失败: {e}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="完整评估脚本 (10月版本逻辑)")

    # 输入路径
    parser.add_argument('--pred', type=str, required=True,
                        help='预测结果目录 (包含patient_id/patient_id_sct_slice_*.npy)')
    parser.add_argument('--gt', type=str, required=True,
                        help='Ground truth目录 (包含patient_id/patient_id_slice_*.npy)')

    # 输出配置
    parser.add_argument('--output', type=str, default='outputs/evaluation_results',
                        help='评估结果输出目录')

    # 评估选项
    parser.add_argument('--full', action='store_true', default=True,
                        help='完整评估 (HU范围+可视化+报告)')
    parser.add_argument('--data-range', type=float, default=2000.0,
                        help='数据范围 (用于PSNR计算)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("CBCT → sCT 完整评估 (10月版本逻辑)")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info(f"预测目录: {args.pred}")
    logger.info(f"GT目录: {args.gt}")
    logger.info(f"完整评估: {'启用' if args.full else '禁用'}")
    logger.info("=" * 60)

    # 查找文件
    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)

    # 递归查找所有NPY文件
    pred_files = sorted(pred_dir.glob("**/*_sct_*.npy"))
    logger.info(f"\n找到 {len(pred_files)} 个预测文件")

    if len(pred_files) == 0:
        logger.error("❌ 错误: 未找到预测文件")
        return

    # 评估循环
    all_metrics = []
    all_hu_range_metrics = []

    for pred_file in tqdm(pred_files, desc="评估中"):
        try:
            # 从预测文件名推导GT文件路径
            # 预测文件: {patient_id}/{patient_id}_sct_slice_{idx}.npy
            # GT文件: {patient_id}/{patient_id}_slice_{idx}.npy
            patient_id = pred_file.parent.name
            gt_filename = pred_file.name.replace('_sct_', '_')
            gt_file = gt_dir / patient_id / gt_filename

            if not gt_file.exists():
                logger.warning(f"未找到GT文件: {gt_file}")
                continue

            # 加载数据
            pred = np.load(pred_file)
            gt = np.load(gt_file)

            # 确保形状匹配
            if pred.shape != gt.shape:
                logger.warning(f"形状不匹配 {pred.shape} vs {gt.shape}: {pred_file.name}")
                continue

            # 计算整体指标
            metrics = evaluate_sample(pred, gt, args.data_range)
            metrics['filename'] = pred_file.name
            metrics['patient_id'] = patient_id
            metrics['pred_mean'] = float(np.mean(pred))
            metrics['gt_mean'] = float(np.mean(gt))
            all_metrics.append(metrics)

            # HU范围评估
            if args.full:
                hu_metrics = evaluate_hu_ranges(pred, gt, args.data_range)
                hu_metrics['filename'] = pred_file.name
                hu_metrics['patient_id'] = patient_id
                all_hu_range_metrics.append(hu_metrics)

        except Exception as e:
            logger.error(f"评估失败 {pred_file.name}: {e}")
            continue

    # 检查结果
    if len(all_metrics) == 0:
        logger.error("❌ 错误: 无有效评估结果")
        return

    logger.info("\n" + "=" * 60)
    logger.info("评估结果汇总")
    logger.info("=" * 60)

    # 计算汇总统计
    summary = {}
    for key in ['MAE', 'RMSE', 'ME', 'PSNR', 'SSIM']:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
        logger.info(f"{key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    # HU范围汇总
    summary_hu_ranges = {}
    if args.full and all_hu_range_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("HU范围评估结果")
        logger.info("=" * 60)

        for range_name in ['Full Range', 'Soft Tissue', 'Low Density', 'High Density']:
            range_metrics = {}
            for metric in ['MAE', 'RMSE', 'ME', 'PSNR', 'SSIM']:
                values = []
                for sample in all_hu_range_metrics:
                    if range_name in sample and metric in sample[range_name]:
                        values.append(sample[range_name][metric])

                if values:
                    range_metrics[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }

            if range_metrics:
                summary_hu_ranges[range_name] = range_metrics
                logger.info(f"\n{range_name}:")
                for metric in ['MAE', 'RMSE', 'PSNR', 'SSIM']:
                    if metric in range_metrics:
                        logger.info(f"  {metric}: {range_metrics[metric]['mean']:.3f} ± {range_metrics[metric]['std']:.3f}")

    # 保存结果
    config = {
        'pred_dir': str(args.pred),
        'gt_dir': str(args.gt),
        'data_range': args.data_range,
        'num_samples': len(all_metrics),
        'timestamp': datetime.now().isoformat(),
    }

    results = {
        'config': config,
        'summary': summary,
        'summary_hu_ranges': summary_hu_ranges,
        'per_sample': all_metrics,
        'per_sample_hu_ranges': all_hu_range_metrics,
    }

    # JSON输出
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ JSON结果已保存: {json_path}")

    # CSV输出 - 整体指标
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['filename', 'patient_id', 'MAE', 'RMSE', 'ME', 'PSNR', 'SSIM']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, '') for k in fieldnames})
    logger.info(f"✓ CSV结果已保存: {csv_path}")

    # CSV输出 - HU范围指标
    if args.full and all_hu_range_metrics:
        csv_hu_path = os.path.join(output_dir, 'evaluation_results_hu_ranges.csv')
        with open(csv_hu_path, 'w', newline='') as f:
            fieldnames = ['filename', 'patient_id', 'hu_range', 'MAE', 'RMSE', 'ME', 'PSNR', 'SSIM', 'pixel_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for sample in all_hu_range_metrics:
                for range_name in ['Full Range', 'Soft Tissue', 'Low Density', 'High Density']:
                    if range_name in sample:
                        row = {
                            'filename': sample['filename'],
                            'patient_id': sample['patient_id'],
                            'hu_range': range_name,
                        }
                        row.update(sample[range_name])
                        writer.writerow(row)
        logger.info(f"✓ HU范围CSV已保存: {csv_hu_path}")

    # 生成可视化
    if args.full:
        logger.info("\n" + "=" * 60)
        logger.info("生成可视化图表")
        logger.info("=" * 60)

        generate_bland_altman_plot(all_metrics, vis_dir, logger)
        generate_boxplot(summary_hu_ranges, vis_dir, logger)
        generate_error_heatmap(all_metrics, vis_dir, logger)
        generate_histograms(all_metrics, vis_dir, logger)

    # 生成Markdown报告
    if args.full:
        logger.info("\n" + "=" * 60)
        logger.info("生成Markdown报告")
        logger.info("=" * 60)
        generate_markdown_report(summary, summary_hu_ranges, config, output_dir, logger)

    logger.info("\n" + "=" * 60)
    logger.info("✅ 评估完成!")
    logger.info(f"结果保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
