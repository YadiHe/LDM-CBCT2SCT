#!/usr/bin/env python3
"""
论文可视化图生成脚本
生成用于论文展示的高质量图像

输出:
- figure_comparison_single.png: 单切片对比 (CBCT, dpCT, sCT, Error)
- figure_multi_slice.png: 多切片展示 (同一患者的不同位置)
- figure_multi_patient.png: 多患者展示
- figure_error_analysis.png: 误差分析图
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 窗宽窗位设置
WINDOWS = {
    'soft_tissue': {'center': 40, 'width': 400},   # [-160, 240] HU
    'bone': {'center': 400, 'width': 1500},        # [-350, 1150] HU
    'lung': {'center': -600, 'width': 1500},       # [-1350, 150] HU
    'full': {'center': 0, 'width': 2000},          # [-1000, 1000] HU
}

def apply_window(image: np.ndarray, window: str = 'soft_tissue') -> np.ndarray:
    """应用窗宽窗位"""
    w = WINDOWS[window]
    vmin = w['center'] - w['width'] / 2
    vmax = w['center'] + w['width'] / 2
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)


def load_patient_data(patient_id: str, pred_dir: Path, gt_dir: Path, cbct_dir: Path,
                      slice_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载患者数据"""
    # 加载预测sCT (格式: {patient_id}_sct_slice_{idx}.npy)
    sct_path = pred_dir / patient_id / f"{patient_id}_sct_slice_{slice_idx:04d}.npy"
    sct = np.load(sct_path)

    # 加载GT dpCT (格式: {patient_id}_slice_{idx}.npy)
    gt_path = gt_dir / patient_id / f"{patient_id}_slice_{slice_idx:04d}.npy"
    gt = np.load(gt_path)

    # 加载CBCT (格式: {patient_id}_slice_{idx}.npy)
    cbct_path = cbct_dir / patient_id / f"{patient_id}_slice_{slice_idx:04d}.npy"
    cbct = np.load(cbct_path)

    return cbct, gt, sct


def get_patient_slice_count(pred_dir: Path, patient_id: str) -> int:
    """获取患者切片数量"""
    patient_path = pred_dir / patient_id
    return len(list(patient_path.glob("*.npy")))


def create_single_comparison_figure(cbct: np.ndarray, gt: np.ndarray, sct: np.ndarray,
                                     output_path: Path, window: str = 'soft_tissue',
                                     roi_box: Optional[Tuple[int, int, int, int]] = None):
    """
    创建单切片对比图

    Layout: CBCT | dpCT (GT) | sCT (Ours) | Error Map
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

    # 应用窗宽窗位
    cbct_win = apply_window(cbct, window)
    gt_win = apply_window(gt, window)
    sct_win = apply_window(sct, window)

    # 计算误差
    error = sct - gt

    # 绘制图像
    titles = ['CBCT', 'dpCT (Ground Truth)', 'sCT (Ours)', 'Error (sCT - dpCT)']
    images = [cbct_win, gt_win, sct_win]

    for i, (ax, img, title) in enumerate(zip(axes[:3], images, titles[:3])):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        # 添加ROI框
        if roi_box is not None:
            x, y, w, h = roi_box
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                      edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)

    # 误差图
    im = axes[3].imshow(error, cmap='RdBu_r', vmin=-100, vmax=100)
    axes[3].set_title(titles[3], fontsize=12, fontweight='bold')
    axes[3].axis('off')

    # 添加colorbar
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('HU', fontsize=10)

    # 计算并显示指标
    mae = np.mean(np.abs(error))
    mask = gt > -900  # 排除空气区域
    mae_tissue = np.mean(np.abs(error[mask]))

    fig.suptitle(f'MAE (full): {mae:.2f} HU | MAE (tissue): {mae_tissue:.2f} HU',
                 fontsize=11, y=0.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_with_zoom(cbct: np.ndarray, gt: np.ndarray, sct: np.ndarray,
                                 output_path: Path, window: str = 'soft_tissue',
                                 zoom_region: Tuple[int, int, int, int] = (180, 180, 150, 150)):
    """
    创建带放大区域的对比图

    Layout:
    Row 1: CBCT | dpCT | sCT | Error
    Row 2: CBCT zoom | dpCT zoom | sCT zoom | Error zoom
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)

    # 应用窗宽窗位
    cbct_win = apply_window(cbct, window)
    gt_win = apply_window(gt, window)
    sct_win = apply_window(sct, window)
    error = sct - gt

    x, y, w, h = zoom_region

    titles = ['CBCT', 'dpCT (GT)', 'sCT (Ours)', 'Error']
    images_full = [cbct_win, gt_win, sct_win]

    # 第一行: 完整图像
    for i, (ax, img, title) in enumerate(zip(axes[0, :3], images_full, titles[:3])):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        # 添加放大区域框
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                  edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    # 误差图
    im = axes[0, 3].imshow(error, cmap='RdBu_r', vmin=-100, vmax=100)
    axes[0, 3].set_title(titles[3], fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                              edgecolor='lime', facecolor='none')
    axes[0, 3].add_patch(rect)

    # 第二行: 放大区域
    zoom_titles = ['CBCT (zoom)', 'dpCT (zoom)', 'sCT (zoom)', 'Error (zoom)']
    for i, (ax, img, title) in enumerate(zip(axes[1, :3], images_full, zoom_titles[:3])):
        ax.imshow(img[y:y+h, x:x+w], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        # 添加绿色边框
        for spine in ax.spines.values():
            spine.set_edgecolor('lime')
            spine.set_linewidth(3)
            spine.set_visible(True)

    axes[1, 3].imshow(error[y:y+h, x:x+w], cmap='RdBu_r', vmin=-100, vmax=100)
    axes[1, 3].set_title(zoom_titles[3], fontsize=11)
    axes[1, 3].axis('off')
    for spine in axes[1, 3].spines.values():
        spine.set_edgecolor('lime')
        spine.set_linewidth(3)
        spine.set_visible(True)

    # 添加colorbar
    cbar = fig.colorbar(im, ax=axes[:, 3], fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('HU', fontsize=10)

    # 计算指标
    mae = np.mean(np.abs(error))
    mask = gt > -900
    mae_tissue = np.mean(np.abs(error[mask]))

    fig.suptitle(f'MAE: {mae:.2f} HU (full) | {mae_tissue:.2f} HU (tissue only)',
                 fontsize=12, y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_multi_slice_figure(pred_dir: Path, gt_dir: Path, cbct_dir: Path,
                               patient_id: str, output_path: Path,
                               window: str = 'soft_tissue'):
    """
    创建多切片展示图 (同一患者的3个不同位置)

    Layout: 3 rows (different slices) x 4 columns (CBCT, dpCT, sCT, Error)
    """
    n_slices = get_patient_slice_count(pred_dir, patient_id)

    # 选择3个代表性切片 (25%, 50%, 75%)
    slice_indices = [int(n_slices * 0.25), int(n_slices * 0.5), int(n_slices * 0.75)]
    slice_labels = ['Superior', 'Middle', 'Inferior']

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=150)

    for row, (slice_idx, label) in enumerate(zip(slice_indices, slice_labels)):
        cbct, gt, sct = load_patient_data(patient_id, pred_dir, gt_dir, cbct_dir, slice_idx)

        cbct_win = apply_window(cbct, window)
        gt_win = apply_window(gt, window)
        sct_win = apply_window(sct, window)
        error = sct - gt

        images = [cbct_win, gt_win, sct_win]

        for col, img in enumerate(images):
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')

            if row == 0:
                titles = ['CBCT', 'dpCT (GT)', 'sCT (Ours)', 'Error']
                axes[row, col].set_title(titles[col], fontsize=12, fontweight='bold')

        # 误差图
        im = axes[row, 3].imshow(error, cmap='RdBu_r', vmin=-100, vmax=100)
        axes[row, 3].axis('off')
        if row == 0:
            axes[row, 3].set_title('Error', fontsize=12, fontweight='bold')

        # 添加slice label
        axes[row, 0].text(-0.15, 0.5, f'{label}\n(slice {slice_idx})',
                          transform=axes[row, 0].transAxes,
                          fontsize=10, va='center', ha='right')

    # 添加colorbar
    cbar = fig.colorbar(im, ax=axes[:, 3], fraction=0.03, pad=0.02)
    cbar.set_label('HU', fontsize=10)

    fig.suptitle(f'Patient {patient_id} - Multi-slice Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_multi_patient_figure(pred_dir: Path, gt_dir: Path, cbct_dir: Path,
                                 patient_ids: List[str], output_path: Path,
                                 window: str = 'soft_tissue'):
    """
    创建多患者展示图

    Layout: N rows (patients) x 4 columns (CBCT, dpCT, sCT, Error)
    """
    n_patients = len(patient_ids)
    fig, axes = plt.subplots(n_patients, 4, figsize=(16, 4 * n_patients), dpi=150)

    if n_patients == 1:
        axes = axes.reshape(1, -1)

    for row, patient_id in enumerate(patient_ids):
        n_slices = get_patient_slice_count(pred_dir, patient_id)
        slice_idx = n_slices // 2  # 选择中间切片

        cbct, gt, sct = load_patient_data(patient_id, pred_dir, gt_dir, cbct_dir, slice_idx)

        cbct_win = apply_window(cbct, window)
        gt_win = apply_window(gt, window)
        sct_win = apply_window(sct, window)
        error = sct - gt

        images = [cbct_win, gt_win, sct_win]

        for col, img in enumerate(images):
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')

            if row == 0:
                titles = ['CBCT', 'dpCT (GT)', 'sCT (Ours)', 'Error']
                axes[row, col].set_title(titles[col], fontsize=12, fontweight='bold')

        # 误差图
        im = axes[row, 3].imshow(error, cmap='RdBu_r', vmin=-100, vmax=100)
        axes[row, 3].axis('off')
        if row == 0:
            axes[row, 3].set_title('Error', fontsize=12, fontweight='bold')

        # 添加patient label和MAE
        mae = np.mean(np.abs(error))
        axes[row, 0].text(-0.15, 0.5, f'Patient {patient_id}\nMAE: {mae:.1f} HU',
                          transform=axes[row, 0].transAxes,
                          fontsize=10, va='center', ha='right')

    # 添加colorbar
    cbar = fig.colorbar(im, ax=axes[:, 3], fraction=0.03, pad=0.02)
    cbar.set_label('HU', fontsize=10)

    fig.suptitle('Multi-patient Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_error_analysis_figure(pred_dir: Path, gt_dir: Path, cbct_dir: Path,
                                  patient_ids: List[str], output_path: Path):
    """
    创建误差分析图

    包含:
    - 整体MAE分布
    - 按区域的MAE分布 (软组织、骨骼、空气)
    """
    # 收集所有数据
    all_errors = {'full': [], 'soft_tissue': [], 'bone': [], 'air': []}
    all_cbct_errors = {'full': [], 'soft_tissue': [], 'bone': [], 'air': []}
    patient_maes = []

    for patient_id in patient_ids:
        n_slices = get_patient_slice_count(pred_dir, patient_id)
        patient_errors = []

        for slice_idx in range(n_slices):
            try:
                cbct, gt, sct = load_patient_data(patient_id, pred_dir, gt_dir, cbct_dir, slice_idx)
                error = np.abs(sct - gt)
                cbct_error = np.abs(cbct - gt)

                # 区域mask
                air_mask = gt < -900
                bone_mask = gt > 200
                soft_tissue_mask = (gt >= -900) & (gt <= 200)

                # 收集误差
                all_errors['full'].append(np.mean(error))
                all_cbct_errors['full'].append(np.mean(cbct_error))

                if np.sum(soft_tissue_mask) > 0:
                    all_errors['soft_tissue'].append(np.mean(error[soft_tissue_mask]))
                    all_cbct_errors['soft_tissue'].append(np.mean(cbct_error[soft_tissue_mask]))

                if np.sum(bone_mask) > 0:
                    all_errors['bone'].append(np.mean(error[bone_mask]))
                    all_cbct_errors['bone'].append(np.mean(cbct_error[bone_mask]))

                if np.sum(air_mask) > 0:
                    all_errors['air'].append(np.mean(error[air_mask]))
                    all_cbct_errors['air'].append(np.mean(cbct_error[air_mask]))

                patient_errors.append(np.mean(error))
            except Exception as e:
                continue

        if patient_errors:
            patient_maes.append((patient_id, np.mean(patient_errors)))

    # 创建图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # 图1: 整体误差分布对比
    ax1 = axes[0, 0]
    data = [all_cbct_errors['full'], all_errors['full']]
    bp = ax1.boxplot(data, labels=['CBCT', 'sCT (Ours)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('MAE (HU)', fontsize=11)
    ax1.set_title('Overall Error Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 添加mean值标注
    means = [np.mean(d) for d in data]
    for i, m in enumerate(means):
        ax1.text(i+1, ax1.get_ylim()[1]*0.95, f'Mean: {m:.1f}',
                 ha='center', fontsize=10, fontweight='bold')

    # 图2: 按区域的误差对比
    ax2 = axes[0, 1]
    regions = ['Soft Tissue', 'Bone', 'Air']
    x = np.arange(len(regions))
    width = 0.35

    cbct_means = [np.mean(all_cbct_errors['soft_tissue']),
                  np.mean(all_cbct_errors['bone']),
                  np.mean(all_cbct_errors['air'])]
    sct_means = [np.mean(all_errors['soft_tissue']),
                 np.mean(all_errors['bone']),
                 np.mean(all_errors['air'])]

    bars1 = ax2.bar(x - width/2, cbct_means, width, label='CBCT', color='lightcoral')
    bars2 = ax2.bar(x + width/2, sct_means, width, label='sCT (Ours)', color='lightgreen')

    ax2.set_ylabel('MAE (HU)', fontsize=11)
    ax2.set_title('Error by Tissue Type', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标注
    for bar, val in zip(bars1, cbct_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, sct_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # 图3: 改善率
    ax3 = axes[1, 0]
    improvement = [(c - s) / c * 100 for c, s in zip(cbct_means, sct_means)]
    colors = ['green' if i > 0 else 'red' for i in improvement]
    bars = ax3.bar(regions, improvement, color=colors, alpha=0.7)
    ax3.set_ylabel('Improvement (%)', fontsize=11)
    ax3.set_title('Improvement over CBCT', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, improvement):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 图4: 每个患者的MAE
    ax4 = axes[1, 1]
    patient_maes_sorted = sorted(patient_maes, key=lambda x: x[1])
    pids = [p[0] for p in patient_maes_sorted]
    maes = [p[1] for p in patient_maes_sorted]

    ax4.barh(pids, maes, color='steelblue', alpha=0.7)
    ax4.set_xlabel('MAE (HU)', fontsize=11)
    ax4.set_title('MAE by Patient', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 添加mean线
    mean_mae = np.mean(maes)
    ax4.axvline(x=mean_mae, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_mae:.1f}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--pred', type=str, required=True, help='Prediction directory')
    parser.add_argument('--gt', type=str, required=True, help='Ground truth directory')
    parser.add_argument('--cbct', type=str, required=True, help='CBCT directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--window', type=str, default='soft_tissue',
                        choices=['soft_tissue', 'bone', 'lung', 'full'],
                        help='Window preset')
    args = parser.parse_args()

    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)
    cbct_dir = Path(args.cbct)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有患者
    patient_ids = sorted([p.name for p in pred_dir.iterdir() if p.is_dir()])
    print(f"Found {len(patient_ids)} patients: {patient_ids}")

    if not patient_ids:
        print("No patients found!")
        return

    # 选择代表性患者
    main_patient = patient_ids[len(patient_ids) // 2]  # 中间的患者
    n_slices = get_patient_slice_count(pred_dir, main_patient)
    main_slice = n_slices // 2  # 中间切片

    print(f"\nUsing patient {main_patient}, slice {main_slice} for single comparison")

    # 加载主要数据
    cbct, gt, sct = load_patient_data(main_patient, pred_dir, gt_dir, cbct_dir, main_slice)

    # 1. 单切片对比图
    print("\n[1/5] Creating single comparison figure...")
    create_single_comparison_figure(
        cbct, gt, sct,
        output_dir / 'figure_comparison_single.png',
        window=args.window
    )

    # 2. 带放大区域的对比图
    print("[2/5] Creating comparison with zoom...")
    # 自动检测感兴趣区域 (找到gt中有组织的区域)
    mask = gt > -500
    if np.sum(mask) > 0:
        y_indices, x_indices = np.where(mask)
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        zoom_region = (max(0, center_x - 75), max(0, center_y - 75), 150, 150)
    else:
        zoom_region = (180, 180, 150, 150)

    create_comparison_with_zoom(
        cbct, gt, sct,
        output_dir / 'figure_comparison_zoom.png',
        window=args.window,
        zoom_region=zoom_region
    )

    # 3. 多切片展示
    print("[3/5] Creating multi-slice figure...")
    create_multi_slice_figure(
        pred_dir, gt_dir, cbct_dir,
        main_patient,
        output_dir / 'figure_multi_slice.png',
        window=args.window
    )

    # 4. 多患者展示
    print("[4/5] Creating multi-patient figure...")
    # 选择最多4个患者
    selected_patients = patient_ids[:min(4, len(patient_ids))]
    create_multi_patient_figure(
        pred_dir, gt_dir, cbct_dir,
        selected_patients,
        output_dir / 'figure_multi_patient.png',
        window=args.window
    )

    # 5. 误差分析图
    print("[5/5] Creating error analysis figure...")
    create_error_analysis_figure(
        pred_dir, gt_dir, cbct_dir,
        patient_ids,
        output_dir / 'figure_error_analysis.png'
    )

    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
