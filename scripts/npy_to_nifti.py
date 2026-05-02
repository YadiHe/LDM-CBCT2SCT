#!/usr/bin/env python
"""
NPY转NIFTI工具
将NPY格式的2D切片文件转换为3D NIFTI格式

支持两种输入格式：
1. 推理输出格式: patient_id/patient_id_sct_slice_XXXX.npy
2. 预处理格式: patient_id/patient_id_slice_XXXX.npy

使用方法:
python npy_to_nifti.py --input <输入目录> --output <输出目录> [选项]

示例:
# 转换推理输出的sCT
python npy_to_nifti.py --input ./inference_output/sct_npy --output ./inference_output/sct_nifti

# 转换预处理后的CT/CBCT
python npy_to_nifti.py --input ./dataset/CT --output ./output/nifti_ct --pattern "*_slice_*.npy"
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from collections import defaultdict
import re
import glob
from tqdm import tqdm

def natural_sort_key(s):
    """用于自然排序的键函数"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def group_npy_files(input_dir, pattern="*.npy"):
    """
    按患者ID和序列分组NPY文件
    返回: {patient_id: {sequence_name: [file_paths]}}
    """
    input_dir = Path(input_dir)
    all_npy_files = list(input_dir.glob(f"**/{pattern}"))
    
    if not all_npy_files:
        print(f"错误: 在 {input_dir} 中未找到匹配 '{pattern}' 的NPY文件")
        return {}
    
    # 按患者ID和序列名分组
    grouped_files = defaultdict(lambda: defaultdict(list))
    
    for file_path in all_npy_files:
        # 确定患者ID
        if len(file_path.parts) > 2 and file_path.parts[-2].isdigit():
            # 如果文件在患者ID命名的子目录中
            patient_id = file_path.parts[-2]
        else:
            # 尝试从文件名中提取患者ID
            match = re.search(r'(\d{6})_slice_', file_path.name)
            if match:
                patient_id = match.group(1)
            else:
                # 如果无法确定患者ID，使用"unknown"
                patient_id = "unknown"
        
        # 确定序列名
        # 支持 patient_id_sct_slice_XXXX.npy 和 patient_id_slice_XXXX.npy 格式
        sequence_name = re.sub(r'_slice_\d+\.npy$', '', file_path.name)
        
        # 对于推理输出，进一步简化序列名
        # 例如: 002085_sct_slice_0000.npy -> sct
        if '_sct' in sequence_name and sequence_name.startswith(str(patient_id)):
            sequence_name = 'sct'
        elif sequence_name.startswith(str(patient_id)):
            # 例如: 002085_slice_0000.npy -> original
            sequence_name = sequence_name.replace(f"{patient_id}_", "")
            if not sequence_name:
                sequence_name = "original"
        
        # 添加到分组中
        grouped_files[patient_id][sequence_name].append(file_path)
    
    # 对每个序列中的文件进行排序
    for patient_id in grouped_files:
        for sequence_name in grouped_files[patient_id]:
            grouped_files[patient_id][sequence_name].sort(key=lambda x: natural_sort_key(x.name))
    
    return grouped_files

def create_nifti_from_npy_files(file_paths, output_path, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    从一组NPY文件创建NIFTI文件
    
    参数:
    - file_paths: NPY文件路径列表
    - output_path: 输出NIFTI文件路径
    - spacing: 体素间距 (x, y, z)
    - origin: 坐标原点 (x, y, z)
    """
    if not file_paths:
        print("错误: 没有提供文件路径")
        return False
    
    # 加载第一个文件以获取切片尺寸
    first_slice = np.load(file_paths[0])
    if first_slice.ndim != 2:
        first_slice = np.squeeze(first_slice)
        if first_slice.ndim != 2:
            print(f"错误: 文件 {file_paths[0]} 不是2D数组")
            return False
    
    # 创建3D体积
    volume_shape = (first_slice.shape[0], first_slice.shape[1], len(file_paths))
    volume = np.zeros(volume_shape, dtype=first_slice.dtype)
    
    # 加载所有切片
    for i, file_path in enumerate(file_paths):
        slice_data = np.load(file_path)
        if slice_data.ndim != 2:
            slice_data = np.squeeze(slice_data)
        
        # 确保所有切片尺寸一致
        if slice_data.shape != first_slice.shape:
            print(f"警告: 文件 {file_path} 的形状 {slice_data.shape} 与第一个切片 {first_slice.shape} 不一致")
            # 调整尺寸或跳过
            continue
        
        # 将切片添加到体积中
        volume[:, :, i] = slice_data
    
    # 创建NIFTI图像
    nifti_img = nib.Nifti1Image(volume, np.eye(4))
    
    # 设置体素间距
    nifti_img.header.set_zooms(spacing)
    
    # 保存NIFTI文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, output_path)
    
    return True

def convert_directory(input_dir, output_dir, pattern="*.npy"):
    """
    将目录中的NPY文件转换为NIFTI格式
    
    参数:
    - input_dir: 输入目录
    - output_dir: 输出目录
    - pattern: 文件匹配模式
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按患者ID和序列分组NPY文件
    grouped_files = group_npy_files(input_dir, pattern)
    
    if not grouped_files:
        print("未找到可处理的文件")
        return
    
    # 处理每个患者的每个序列
    total_patients = len(grouped_files)
    print(f"找到 {total_patients} 个患者的数据")
    
    for patient_id, sequences in grouped_files.items():
        print(f"处理患者 {patient_id}，共 {len(sequences)} 个序列")
        
        # 为患者创建输出目录
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        for sequence_name, file_paths in tqdm(sequences.items(), desc=f"患者 {patient_id} 的序列"):
            # 创建输出NIFTI文件路径
            output_path = patient_output_dir / f"{sequence_name}.nii.gz"
            
            # 创建NIFTI文件
            success = create_nifti_from_npy_files(file_paths, output_path)
            
            if success:
                print(f"  已创建: {output_path} (从 {len(file_paths)} 个切片)")
            else:
                print(f"  创建失败: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='NPY转NIFTI工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换推理输出的sCT
  python npy_to_nifti.py -i ./output/generated_sct -o ./output/nifti_sct
  
  # 转换预处理后的CT数据
  python npy_to_nifti.py -i ./dataset/CT -o ./output/nifti_ct
  
  # 指定特定文件模式
  python npy_to_nifti.py -i ./data -o ./output -p "*sct*.npy"
        """
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='包含NPY文件的输入目录（患者子目录结构）')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='NIFTI文件输出目录')
    parser.add_argument('--pattern', '-p', type=str, default="*.npy",
                       help='NPY文件匹配模式 (默认: *.npy)')
    parser.add_argument('--spacing', '-s', type=float, nargs=3, 
                       default=[1.0, 1.0, 3.0],
                       help='体素间距 (x y z)，默认: 1.0 1.0 3.0')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NPY to NIFTI Conversion Tool")
    print("="*60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"文件模式: {args.pattern}")
    print(f"体素间距: {args.spacing}")
    print("="*60)
    
    convert_directory(args.input, args.output, args.pattern)
    
    print("="*60)
    print("✓ 转换完成!")
    print("="*60)

if __name__ == "__main__":
    main()
