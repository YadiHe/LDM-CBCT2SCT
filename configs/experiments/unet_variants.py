"""
UNet 架构对比实验配置

实验目的：比较不同条件融合策略的效果
- Concatenation: 早期级联，简单高效
- Skip: Skip-connection融合，保留细节
- CrossAttention: 交叉注意力，灵活选择
- ControlPACA: ControlNet + PACA，精细控制
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class UNetExperiment:
    """单个UNet实验配置"""
    name: str
    unet_type: str
    description: str
    model_class: str  # 模型类名
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


# ============================================================================
# 实验定义
# ============================================================================

UNET_EXPERIMENTS = {
    "concatenation": UNetExperiment(
        name="UNet-Concatenation",
        unet_type="concatenation",
        description="早期级联：CBCT和噪声潜变量在输入层concat",
        model_class="UNetConcatenation",
        extra_params={
            "in_channels": 3,  # 实际会变成6 (3+3)
        }
    ),

    "skip": UNetExperiment(
        name="UNet-Skip",
        unet_type="skip",
        description="Skip-connection：条件通过独立编码器，在上采样时融合",
        model_class="UNetSkip",
        extra_params={}
    ),

    "cross_attention": UNetExperiment(
        name="UNet-CrossAttention",
        unet_type="cross_attention",
        description="交叉注意力：在每个block中使用cross-attention融合条件",
        model_class="UNetCrossAttention",
        extra_params={}
    ),

    "control_paca": UNetExperiment(
        name="UNet-ControlPACA",
        unet_type="control_paca",
        description="ControlNet + PACA：精细控制生成过程",
        model_class="UNetControlPACA",
        extra_params={
            "use_paca": True,
        }
    ),
}


# ============================================================================
# 实验运行配置
# ============================================================================

EXPERIMENT_CONFIG = {
    # 共享训练参数
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "early_stopping": 30,

    # 评估设置
    "eval_ddim_steps": 40,
    "eval_samples": 100,  # 评估样本数

    # 输出目录模板
    "output_dir_template": "experiments/unet_variants/{experiment_name}",
}


def get_experiment(name: str) -> UNetExperiment:
    """获取指定实验配置"""
    if name not in UNET_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(UNET_EXPERIMENTS.keys())}")
    return UNET_EXPERIMENTS[name]


def list_experiments() -> None:
    """打印所有可用实验"""
    print("=" * 60)
    print("UNet 架构对比实验")
    print("=" * 60)
    for key, exp in UNET_EXPERIMENTS.items():
        print(f"\n[{key}] {exp.name}")
        print(f"    类型: {exp.unet_type}")
        print(f"    说明: {exp.description}")
        print(f"    模型类: {exp.model_class}")
