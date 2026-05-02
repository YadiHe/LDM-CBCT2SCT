#!/usr/bin/env python
"""
统一推理入口脚本

用法:
    # 基础推理
    python scripts/infer.py --vae checkpoints/vae.pth --unet checkpoints/unet.pth --input data/test

    # 使用 DDIM 快速采样
    python scripts/infer.py --vae vae.pth --unet unet.pth --input test/ --ddim-steps 40

    # 使用 CFG
    python scripts/infer.py --vae vae.pth --unet unet.pth --input test/ --use-cfg --cfg-scale 7.5

    # 批量推理并保存为 NIfTI
    python scripts/infer.py --vae vae.pth --unet unet.pth --input test/ --output results/ --save-nifti
"""
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="统一推理脚本")

    # 模型路径
    parser.add_argument('--vae', type=str, required=True,
                        help='VAE模型路径')
    parser.add_argument('--unet', type=str, required=True,
                        help='UNet模型路径')

    # 输入输出
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据目录或manifest文件')
    parser.add_argument('--output', type=str, default='outputs/inference',
                        help='输出目录')

    # 采样配置
    parser.add_argument('--method', type=str, default='ddim',
                        choices=['ddpm', 'ddim'],
                        help='采样方法')
    parser.add_argument('--ddim-steps', type=int, default=40,
                        help='DDIM采样步数（10月用40步）')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                        help='DDIM随机性参数 (0=确定性)')
    parser.add_argument('--schedule', type=str, default='linear',
                        choices=['linear', 'quadratic', 'power'],
                        help='时间步调度策略')

    # CBCT初始化配置（训练时未使用，默认禁用以匹配训练）
    parser.add_argument('--use-cbct-init', action='store_true', default=False,
                        help='使用CBCT信号初始化（训练时未使用）')
    parser.add_argument('--alpha-a', type=float, default=0.5,
                        help='CBCT初始化混合比例')

    # CFG配置
    parser.add_argument('--use-cfg', action='store_true',
                        help='启用Classifier-Free Guidance')
    parser.add_argument('--cfg-scale', type=float, default=7.5,
                        help='CFG强度')

    # 可视化配置（10月有）
    parser.add_argument('--save-vis', action='store_true', default=True,
                        help='保存可视化对比图')
    parser.add_argument('--vis-freq', type=int, default=10,
                        help='每N个样本保存一次可视化')

    # 输出格式
    parser.add_argument('--save-npy', action='store_true', default=True,
                        help='保存NPY格式')
    parser.add_argument('--save-nifti', action='store_true',
                        help='保存NIfTI格式')

    # 其他
    parser.add_argument('--batch-size', type=int, default=1,
                        help='推理批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    return parser.parse_args()


def postprocess(tensor, mode="linear"):
    """
    后处理：将模型输出转换回HU值

    Args:
        tensor: 归一化的tensor [-1, 1]
        mode: 预处理模式

    Returns:
        HU值 numpy array
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.cpu().numpy()
    else:
        arr = tensor

    if mode == "linear":
        # 逆变换: HU = normalized * 1250 + 250
        hu = arr * 1250 + 250
    else:
        hu = arr * 1000  # 简单缩放

    # 裁剪到合理范围
    hu = np.clip(hu, -1000, 1500)
    return hu


def load_models(args):
    """加载模型"""
    from models.vae import load_vae
    from models.diffusion import Diffusion
    from models.unetConditional import UNetSkip, UNetConcatenation, UNetCrossAttention

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"加载 VAE: {args.vae}")
    vae = load_vae(args.vae, trainable=False)
    vae = vae.to(device)
    vae.eval()

    print(f"加载 UNet: {args.unet}")

    # 自动检测架构类型
    checkpoint = torch.load(args.unet, map_location='cpu')

    # 检查checkpoint格式：可能是直接的state_dict或包含训练状态的完整checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整checkpoint格式 (unet_last_ep199.pth)
        print(f"检测到完整checkpoint格式 (epoch {checkpoint.get('epoch', '?')})")
        state_dict = checkpoint['model_state_dict']
    else:
        # 直接的state_dict格式 (unet_best.pth)
        state_dict = checkpoint

    # 根据state_dict的keys判断架构
    # 优先级: cross_attention > skip > concatenation
    has_cond_init = 'cond_init_conv.weight' in state_dict
    has_cross_attn = any('cross_attention' in k for k in state_dict.keys())

    if has_cond_init and has_cross_attn:
        # Cross Attention架构有两个特征
        arch_type = 'cross_attention'
        base_channels = state_dict['init_conv.weight'].shape[0]
    elif has_cond_init:
        # Skip架构只有cond_init_conv
        arch_type = 'skip'
        base_channels = state_dict['init_conv.weight'].shape[0]
    else:
        # Concatenation架构没有这两个特征
        arch_type = 'concatenation'
        base_channels = state_dict['init_conv.weight'].shape[0]

    print(f"检测到架构: {arch_type}, base_channels: {base_channels}")

    # 初始化对应的UNet
    if arch_type == 'skip':
        unet = UNetSkip(in_channels=3, out_channels=3,
                       base_channels=base_channels, dropout_rate=0.1)
    elif arch_type == 'cross_attention':
        unet = UNetCrossAttention(in_channels=3, out_channels=3,
                                  base_channels=base_channels, dropout_rate=0.1)
    else:  # concatenation
        unet = UNetConcatenation(in_channels=3, out_channels=3,
                                base_channels=base_channels, dropout_rate=0.1)

    # 加载权重
    unet.load_state_dict(state_dict)
    unet = unet.to(device)
    unet.eval()

    print(f"✓ UNet加载成功")

    diffusion = Diffusion(device=device, timesteps=1000)

    return vae, unet, diffusion, device


def get_timesteps(total_steps, num_steps, schedule):
    """生成时间步序列"""
    if schedule == "linear":
        return list(np.linspace(0, total_steps - 1, num_steps, dtype=int)[::-1])
    elif schedule == "quadratic":
        t = np.linspace(0, 1, num_steps) ** 2
        return list((t * (total_steps - 1)).astype(int)[::-1])
    elif schedule == "power":
        coarse = num_steps // 3
        fine = num_steps - coarse
        coarse_t = np.linspace(total_steps - 1, total_steps * 0.3, coarse)
        fine_t = np.linspace(total_steps * 0.3, 0, fine)
        return list(np.concatenate([coarse_t, fine_t]).astype(int))
    else:
        return list(range(total_steps - 1, -1, total_steps // num_steps))


@torch.no_grad()
def ddim_sample(unet, vae, diffusion, condition, timesteps, eta, device, cfg_scale=1.0, use_cbct_init=True, alpha_a=0.5):
    """
    DDIM采样（兼容10月配置）

    Args:
        unet: UNet模型
        vae: VAE模型
        diffusion: Diffusion对象
        condition: 条件图像 (CBCT)
        timesteps: 时间步序列
        eta: 随机性参数
        device: 设备
        cfg_scale: CFG强度
        use_cbct_init: 是否使用CBCT初始化（10月用的）
        alpha_a: CBCT初始化混合比例

    Returns:
        生成的sCT图像
    """
    # 编码条件
    with torch.no_grad():
        cond_mu, cond_logvar = vae.encode(condition)
        cond_latent = vae.reparameterize(cond_mu, cond_logvar)

    # 初始化噪声（10月的CBCT初始化）
    b, c, h, w = cond_latent.shape

    if use_cbct_init:
        # CBCT初始化：混合CBCT信号和噪声
        t0 = timesteps[0]
        alpha_bar_t0 = diffusion.alpha_cumprod[t0]
        alpha_eff = alpha_a * alpha_bar_t0
        s_alpha = torch.sqrt(alpha_eff)
        s_noise = torch.sqrt(1.0 - alpha_eff)
        noise = torch.randn(b, 3, h, w, device=device)
        x = s_alpha * cond_latent + s_noise * noise
    else:
        # 纯随机初始化
        x = torch.randn(b, 3, h, w, device=device)

    alphas = diffusion.alpha.to(device)
    alpha_bars = diffusion.alpha_cumprod.to(device)

    for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling", leave=False)):
        t_tensor = torch.tensor([t], device=device).long()

        # 预测噪声
        if cfg_scale > 1.0:
            # CFG: 同时预测有条件和无条件
            noise_cond = unet(x, cond_latent, t_tensor)
            noise_uncond = unet(x, torch.zeros_like(cond_latent), t_tensor)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = unet(x, cond_latent, t_tensor)

        # DDIM更新
        alpha_bar_t = alpha_bars[t]
        if i + 1 < len(timesteps):
            alpha_bar_t_prev = alpha_bars[timesteps[i + 1]]
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)

        # 预测x0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        # 注意：训练时没有clamp，为了匹配训练逻辑，这里也不clamp
        # x0_pred = torch.clamp(x0_pred, -1, 1)

        # 计算方向
        sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * \
                  torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)

        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * noise_pred

        # 更新x
        if i + 1 < len(timesteps):
            noise = torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt + sigma_t * noise
        else:
            x = x0_pred

    # 解码
    output = vae.decode(x)
    return output


def save_visualization(cbct, sct, ct, save_path, window_center=250, window_width=2500):
    """
    保存可视化对比图（10月的功能）

    Args:
        cbct: CBCT图像 (HU值)
        sct: 生成的sCT图像 (HU值)
        ct: Ground truth CT图像 (HU值)
        save_path: 保存路径
        window_center: 窗位
        window_width: 窗宽
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 窗位窗宽
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2

    # CBCT输入
    im0 = axes[0].imshow(cbct, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'CBCT Input\nHU: [{cbct.min():.0f}, {cbct.max():.0f}]')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # 生成的sCT
    im1 = axes[1].imshow(sct, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Generated sCT\nHU: [{sct.min():.0f}, {sct.max():.0f}]')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Ground Truth CT
    im2 = axes[2].imshow(ct, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Ground Truth CT\nHU: [{ct.min():.0f}, {ct.max():.0f}]')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    print("=" * 60)
    print("CBCT → sCT 推理脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 训练时的标准配置（来自 scripts/train_experiment.py）
    # 注意：如果修改训练代码，请同步更新这里的配置
    TRAINING_CONFIG = {
        'ddim_steps': 40,           # train_experiment.py:490
        'use_cbct_init': False,     # train_experiment.py:493 (使用 torch.randn_like 纯随机初始化)
        'use_cfg': False,           # 训练时未使用CFG
        'source': 'scripts/train_experiment.py (generate_predictions函数)'
    }

    # 当前推理配置
    print("\n【推理配置】")
    print(f"  采样方法: {args.method}")
    print(f"  DDIM步数: {args.ddim_steps}")
    print(f"  CBCT初始化: {'启用 (alpha={})'.format(args.alpha_a) if args.use_cbct_init else '禁用'}")
    print(f"  CFG: {'启用 (scale={})'.format(args.cfg_scale) if args.use_cfg else '禁用'}")

    # 训练配置信息
    print("\n【训练配置参考】")
    print(f"  来源: {TRAINING_CONFIG['source']}")
    print(f"  DDIM步数: {TRAINING_CONFIG['ddim_steps']}")
    print(f"  CBCT初始化: {'启用' if TRAINING_CONFIG['use_cbct_init'] else '禁用'}")
    print(f"  CFG: {'启用' if TRAINING_CONFIG['use_cfg'] else '禁用'}")

    # 训练-推理一致性检查
    print("\n【一致性检查】")
    consistency_issues = []

    if args.ddim_steps != TRAINING_CONFIG['ddim_steps']:
        msg = f"DDIM步数: 训练={TRAINING_CONFIG['ddim_steps']}, 推理={args.ddim_steps}"
        print(f"  ⚠️  {msg}")
        consistency_issues.append(msg)
    else:
        print(f"  ✓ DDIM步数一致: {args.ddim_steps}")

    if args.use_cbct_init != TRAINING_CONFIG['use_cbct_init']:
        train_status = '启用' if TRAINING_CONFIG['use_cbct_init'] else '禁用'
        infer_status = '启用' if args.use_cbct_init else '禁用'
        msg = f"CBCT初始化: 训练={train_status}, 推理={infer_status}"
        print(f"  ⚠️  {msg}")
        consistency_issues.append(msg)
    else:
        print(f"  ✓ CBCT初始化一致: {'启用' if args.use_cbct_init else '禁用'}")

    if args.use_cfg != TRAINING_CONFIG['use_cfg']:
        train_status = '启用' if TRAINING_CONFIG['use_cfg'] else '禁用'
        infer_status = '启用' if args.use_cfg else '禁用'
        msg = f"CFG: 训练={train_status}, 推理={infer_status}"
        print(f"  ⚠️  {msg}")
        consistency_issues.append(msg)
    else:
        print(f"  ✓ CFG一致: {'启用' if args.use_cfg else '禁用'}")

    if not consistency_issues:
        print(f"\n  ✅ 配置完全一致！生成质量应该最优")
    else:
        print(f"\n  ⚠️  发现 {len(consistency_issues)} 个不一致项:")
        for issue in consistency_issues:
            print(f"     - {issue}")
        print(f"  💡 这可能导致生成质量下降，建议调整推理参数以匹配训练配置")

    print("=" * 60)

    # 创建输出目录
    output_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    vae, unet, diffusion, device = load_models(args)

    # 生成时间步
    timesteps = get_timesteps(1000, args.ddim_steps, args.schedule)

    # 加载测试数据
    from utils.dataset_256 import PairedCTCBCTDatasetNPY256
    from torch.utils.data import DataLoader

    # 这里简化处理，实际应该根据input类型加载数据
    print(f"\n从 {args.input} 加载数据...")

    # 假设input是manifest文件
    if args.input.endswith('.csv'):
        dataset = PairedCTCBCTDatasetNPY256(
            manifest_csv=args.input,
            split='test',
            target_size=(256, 256),
        )
    else:
        # 如果是目录，需要创建临时manifest或直接加载
        print("注意: 目录输入需要manifest文件，请使用 --input manifest.csv")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"共 {len(dataset)} 个样本")

    # 推理循环
    results = []
    for batch_idx, (ct_gt, cbct) in enumerate(tqdm(loader, desc="推理中")):
        cbct = cbct.to(device)
        ct_gt = ct_gt.to(device) if ct_gt is not None else None

        # 获取文件信息（从manifest中提取patient_id）
        # batch_size可能>1，需要遍历
        batch_size = cbct.shape[0]

        # 采样（加入10月的CBCT初始化参数）
        if args.method == 'ddim':
            output = ddim_sample(
                unet, vae, diffusion, cbct, timesteps,
                eta=args.ddim_eta, device=device,
                cfg_scale=args.cfg_scale if args.use_cfg else 1.0,
                use_cbct_init=args.use_cbct_init,
                alpha_a=args.alpha_a
            )
        else:
            # DDPM (完整1000步)
            output = ddim_sample(
                unet, vae, diffusion, cbct,
                list(range(999, -1, -1)),
                eta=1.0, device=device, cfg_scale=1.0,
                use_cbct_init=args.use_cbct_init,
                alpha_a=args.alpha_a
            )

        # 处理batch中的每个样本
        for i in range(batch_size):
            global_idx = batch_idx * args.batch_size + i

            # 从dataset获取文件路径信息
            row = dataset.df.iloc[global_idx]
            ct_path = row['ct_path']
            # 从路径提取patient_id: .../CT/002243/002243_slice_0000.npy
            patient_id = ct_path.split('/')[-2]
            slice_idx = row['slice_idx']

            # 后处理
            output_hu = postprocess(output[i].squeeze(0))
            ct_hu = postprocess(ct_gt[i].squeeze(0)) if ct_gt is not None else None
            cbct_hu = postprocess(cbct[i].squeeze(0))

            # 保存到患者子目录
            if args.save_npy:
                patient_dir = os.path.join(output_dir, patient_id)
                os.makedirs(patient_dir, exist_ok=True)
                # 文件名格式: {patient_id}_sct_slice_{idx}.npy
                save_path = os.path.join(patient_dir, f"{patient_id}_sct_slice_{slice_idx:04d}.npy")
                np.save(save_path, output_hu)

            # 保存可视化（10月的功能）
            if args.save_vis and (global_idx % args.vis_freq == 0) and ct_hu is not None:
                vis_dir = os.path.join(output_dir, '../visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"comparison_{global_idx:04d}.png")
                save_visualization(cbct_hu, output_hu, ct_hu, vis_path)

            results.append({
                'idx': global_idx,
                'patient_id': patient_id,
                'slice_idx': slice_idx,
                'output': output_hu,
                'gt': ct_hu
            })

    print(f"\n✓ 推理完成，结果保存到: {output_dir}")
    print(f"共生成 {len(results)} 个样本")


if __name__ == '__main__':
    main()
