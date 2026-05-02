#!/usr/bin/env python
"""
专为256×256数据定制的数据集类
适用于主实验：使用预处理阶段生成的256×256 NPY文件
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import random


class PairedCTCBCTDatasetNPY256(Dataset):
    """
    专为256×256数据设计的配对CT-CBCT数据集
    保持原始分辨率，不强制resize
    """
    
    def __init__(self, manifest_csv: str, split: str, 
                 target_size: tuple = (256, 256),
                 augmentation=None, 
                 preprocess="linear"):
        """
        Args:
            manifest_csv: manifest文件路径
            split: 数据集划分 ('train', 'val', 'test')
            target_size: 目标图像大小 (width, height)
            augmentation: 数据增强参数
            preprocess: 预处理方式 ('linear' 或 'tanh')
        """
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.target_size = target_size
        self.preprocess = preprocess
        
        print(f"Loading {split} dataset: {len(self.df)} samples")
        print(f"Target size: {target_size}")
        print(f"Using preprocessing: {preprocess}")
        
        # 基础变换（保持原始尺寸或调整到目标尺寸）
        if target_size == (256, 256):
            # 256×256数据，只需要padding处理边界
            self.base_transform = transforms.Compose([
                transforms.Pad((0, 0, 0, 0), fill=-1),  # 不需要padding
            ])
        else:
            # 需要resize到其他尺寸
            self.base_transform = transforms.Compose([
                transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
            ])
        
        # 数据增强
        if augmentation is not None:
            degrees = augmentation.get('degrees', 0)
            translate = augmentation.get('translate', None)
            scale = augmentation.get('scale', None)
            shear = augmentation.get('shear', None)
            
            self.augmentation_transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    fill=-1
                ),
            ])
        else:
            self.augmentation_transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载CT和CBCT数据
        ct = np.load(row['ct_path']).astype(np.float32)
        
        # 优先使用256×256路径（主实验用预处理生成的256数据）
        if 'cbct_256_path' in row and row['cbct_256_path']:
            cbct = np.load(row['cbct_256_path']).astype(np.float32)
        elif 'cbct_512_path' in row and row['cbct_512_path']:
            cbct = np.load(row['cbct_512_path']).astype(np.float32)
        elif 'cbct_490_path' in row and row['cbct_490_path']:
            cbct = np.load(row['cbct_490_path']).astype(np.float32)
        else:
            raise ValueError(f"未找到有效的CBCT路径，行: {idx}")
        
        # 预处理（基于用户指定的HU范围(-1000, 1500)）
        if self.preprocess == "linear":
            # 基于(-1000, 1500) HU范围的归一化到[-1, 1]
            # 中点: 250, 半径: 1250  =>  (HU - 250) / 1250
            ct = (ct - 250.0) / 1250.0      # (-1000,1500) -> (-1,1)
            cbct = (cbct - 250.0) / 1250.0  # (-1000,1500) -> (-1,1)
        elif self.preprocess == "tanh":
            # 保持原有tanh方式
            ct = np.tanh(ct / 150.0)
            cbct = np.tanh(cbct / 150.0)
        
        # 转换为tensor
        ct = torch.from_numpy(ct).unsqueeze(0)
        cbct = torch.from_numpy(cbct).unsqueeze(0)
        
        # 应用基础变换
        if self.base_transform:
            ct = self.base_transform(ct)
            cbct = self.base_transform(cbct)
        
        # 应用数据增强（同步变换CT和CBCT）
        if self.augmentation_transform:
            # 设置相同的随机种子确保同步变换
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            ct = self.augmentation_transform(ct)
            
            torch.manual_seed(seed)
            cbct = self.augmentation_transform(cbct)
        
        return ct, cbct


class CTDatasetNPY256(Dataset):
    """
    专为256×256数据设计的单模态CT数据集（用于VAE训练）
    """
    
    def __init__(self, manifest_csv: str, split: str,
                 target_size: tuple = (256, 256),
                 augmentation=None,
                 preprocess="linear"):
        """
        Args:
            manifest_csv: manifest文件路径
            split: 数据集划分
            target_size: 目标图像大小
            augmentation: 数据增强参数
            preprocess: 预处理方式
        """
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.target_size = target_size
        self.preprocess = preprocess
        
        print(f"Loading {split} dataset: {len(self.df)} CT samples")
        print(f"Target size: {target_size}")
        print(f"Using preprocessing: {preprocess}")
        
        # 基础变换
        if target_size == (256, 256):
            self.base_transform = transforms.Compose([
                transforms.Pad((0, 0, 0, 0), fill=-1),
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
            ])
        
        # 数据增强
        if augmentation is not None:
            degrees = augmentation.get('degrees', 0)
            translate = augmentation.get('translate', None)
            scale = augmentation.get('scale', None)
            shear = augmentation.get('shear', None)
            
            self.augmentation_transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    fill=-1
                ),
            ])
        else:
            self.augmentation_transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载CT数据
        ct = np.load(row['ct_path']).astype(np.float32)
        
        # 预处理（基于用户指定的HU范围(-1000, 1500)）
        if self.preprocess == "linear":
            # 基于(-1000, 1500) HU范围的归一化到[-1, 1]
            ct = (ct - 250.0) / 1250.0      # (-1000,1500) -> (-1,1)
        elif self.preprocess == "tanh":
            # 保持原有tanh方式
            ct = np.tanh(ct / 150.0)
        
        # 转换为tensor
        ct = torch.from_numpy(ct).unsqueeze(0)
        
        # 应用变换
        if self.base_transform:
            ct = self.base_transform(ct)
        
        if self.augmentation_transform:
            ct = self.augmentation_transform(ct)
        
        return ct


def get_dataloaders_256(manifest_path, batch_size, num_workers, 
                       dataset_class=PairedCTCBCTDatasetNPY256,
                       target_size=(256, 256),
                       train_size=None, val_size=None, test_size=None,
                       augmentation=None, preprocess="linear"):
    """
    专为256×256数据设计的数据加载器
    
    Args:
        manifest_path: manifest.csv文件路径
        batch_size: 批处理大小
        num_workers: 数据加载线程数
        dataset_class: 数据集类（默认PairedCTCBCTDatasetNPY256）
        target_size: 目标图像大小（默认256×256）
        train_size: 训练集大小限制（None表示使用全部）
        val_size: 验证集大小限制（None表示使用全部）
        test_size: 测试集大小限制（None表示使用全部）
        augmentation: 数据增强参数字典
        preprocess: 预处理方式（'linear' 或 'tanh'）
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, Subset
    
    # 创建数据集
    train_dataset = dataset_class(manifest_path, 'train', target_size, augmentation, preprocess)
    val_dataset = dataset_class(manifest_path, 'val', target_size, None, preprocess)  # 验证集不做增强
    test_dataset = dataset_class(manifest_path, 'test', target_size, None, preprocess)  # 测试集不做增强
    
    # 可选的数据子集
    if train_size is not None and train_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_size].tolist()
        train_dataset = Subset(train_dataset, indices)

    if val_size is not None and val_size < len(val_dataset):
        indices = torch.randperm(len(val_dataset))[:val_size].tolist()
        val_dataset = Subset(val_dataset, indices)

    if test_size is not None and test_size < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:test_size].tolist()
        test_dataset = Subset(test_dataset, indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
