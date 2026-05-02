"""
采样策略对比实验配置

实验目的：比较不同采样方法和步数对生成质量的影响
- DDPM vs DDIM
- 不同DDIM步数（10, 20, 40, 100）
- 不同时间步调度策略
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SamplingExperiment:
    """采样实验配置"""
    name: str
    method: str  # "ddpm" 或 "ddim"
    steps: int
    eta: float  # DDIM的随机性参数，0=确定性，1=等同DDPM
    schedule: str  # "linear", "quadratic", "power"
    description: str


# ============================================================================
# 实验定义
# ============================================================================

SAMPLING_EXPERIMENTS = {
    # DDPM 基线
    "ddpm_1000": SamplingExperiment(
        name="DDPM 1000步",
        method="ddpm",
        steps=1000,
        eta=1.0,
        schedule="linear",
        description="完整DDPM采样，最高质量但最慢"
    ),

    # DDIM 不同步数
    "ddim_10": SamplingExperiment(
        name="DDIM 10步",
        method="ddim",
        steps=10,
        eta=0.0,
        schedule="linear",
        description="极快采样，质量可能下降"
    ),

    "ddim_20": SamplingExperiment(
        name="DDIM 20步",
        method="ddim",
        steps=20,
        eta=0.0,
        schedule="linear",
        description="快速采样，质量-速度平衡"
    ),

    "ddim_40": SamplingExperiment(
        name="DDIM 40步 (推荐)",
        method="ddim",
        steps=40,
        eta=0.0,
        schedule="linear",
        description="推荐设置，质量接近DDPM"
    ),

    "ddim_50": SamplingExperiment(
        name="DDIM 50步",
        method="ddim",
        steps=50,
        eta=0.0,
        schedule="linear",
        description="高质量采样"
    ),

    "ddim_100": SamplingExperiment(
        name="DDIM 100步",
        method="ddim",
        steps=100,
        eta=0.0,
        schedule="linear",
        description="接近DDPM质量"
    ),

    # 不同调度策略
    "ddim_40_quadratic": SamplingExperiment(
        name="DDIM 40步 (二次调度)",
        method="ddim",
        steps=40,
        eta=0.0,
        schedule="quadratic",
        description="二次时间步调度，前期更密集"
    ),

    "ddim_40_power": SamplingExperiment(
        name="DDIM 40步 (幂次调度)",
        method="ddim",
        steps=40,
        eta=0.0,
        schedule="power",
        description="幂次混合调度，粗细结合"
    ),

    # DDIM 带随机性
    "ddim_40_stochastic": SamplingExperiment(
        name="DDIM 40步 (随机)",
        method="ddim",
        steps=40,
        eta=0.5,
        schedule="linear",
        description="半随机采样，增加多样性"
    ),
}


# ============================================================================
# 实验组
# ============================================================================

EXPERIMENT_GROUPS = {
    "steps_comparison": [
        "ddim_10", "ddim_20", "ddim_40", "ddim_50", "ddim_100", "ddpm_1000"
    ],
    "schedule_comparison": [
        "ddim_40", "ddim_40_quadratic", "ddim_40_power"
    ],
    "quality_vs_speed": [
        "ddim_10", "ddim_40", "ddpm_1000"
    ],
}


# ============================================================================
# 工具函数
# ============================================================================

def get_timesteps(total_steps: int, num_steps: int, schedule: str = "linear") -> List[int]:
    """
    根据调度策略生成时间步序列

    Args:
        total_steps: 总时间步数（通常1000）
        num_steps: 采样步数
        schedule: 调度策略

    Returns:
        时间步列表
    """
    import numpy as np

    if schedule == "linear":
        # 均匀分布
        return list(np.linspace(0, total_steps - 1, num_steps, dtype=int)[::-1])

    elif schedule == "quadratic":
        # 二次分布，前期更密集
        t = np.linspace(0, 1, num_steps)
        t = t ** 2
        return list((t * (total_steps - 1)).astype(int)[::-1])

    elif schedule == "power":
        # 幂次混合：粗糙阶段稀疏，精细阶段密集
        # 前1/3步用于粗糙去噪，后2/3用于精细调整
        coarse_steps = num_steps // 3
        fine_steps = num_steps - coarse_steps

        coarse = np.linspace(total_steps - 1, total_steps * 0.3, coarse_steps)
        fine = np.linspace(total_steps * 0.3, 0, fine_steps)

        return list(np.concatenate([coarse, fine]).astype(int))

    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def list_experiments() -> None:
    """打印所有可用实验"""
    print("=" * 60)
    print("采样策略对比实验")
    print("=" * 60)

    for key, exp in SAMPLING_EXPERIMENTS.items():
        print(f"\n[{key}] {exp.name}")
        print(f"    方法: {exp.method.upper()}")
        print(f"    步数: {exp.steps}")
        print(f"    调度: {exp.schedule}")
        if exp.method == "ddim":
            print(f"    Eta: {exp.eta} ({'确定性' if exp.eta == 0 else '随机'})")
        print(f"    说明: {exp.description}")

    print("\n" + "=" * 60)
    print("实验组")
    print("=" * 60)
    for group_name, experiments in EXPERIMENT_GROUPS.items():
        print(f"\n{group_name}:")
        for exp_name in experiments:
            print(f"  - {exp_name}")
