"""
Classifier-Free Guidance 对比实验配置

实验目的：比较有无CFG以及不同guidance scale的效果
"""
from dataclasses import dataclass
from typing import List

@dataclass
class CFGExperiment:
    """CFG实验配置"""
    name: str
    use_cfg: bool
    cfg_scale: float
    cfg_dropout_rate: float
    description: str


# ============================================================================
# 实验定义
# ============================================================================

CFG_EXPERIMENTS = {
    "no_cfg": CFGExperiment(
        name="Baseline (No CFG)",
        use_cfg=False,
        cfg_scale=1.0,
        cfg_dropout_rate=0.0,
        description="基线模型，无Classifier-Free Guidance"
    ),

    "cfg_scale_3": CFGExperiment(
        name="CFG Scale 3.0",
        use_cfg=True,
        cfg_scale=3.0,
        cfg_dropout_rate=0.15,
        description="较弱的guidance，保持多样性"
    ),

    "cfg_scale_5": CFGExperiment(
        name="CFG Scale 5.0",
        use_cfg=True,
        cfg_scale=5.0,
        cfg_dropout_rate=0.15,
        description="中等guidance强度"
    ),

    "cfg_scale_7.5": CFGExperiment(
        name="CFG Scale 7.5 (推荐)",
        use_cfg=True,
        cfg_scale=7.5,
        cfg_dropout_rate=0.15,
        description="常用的guidance强度，平衡质量和多样性"
    ),

    "cfg_scale_10": CFGExperiment(
        name="CFG Scale 10.0",
        use_cfg=True,
        cfg_scale=10.0,
        cfg_dropout_rate=0.15,
        description="较强的guidance，更接近条件"
    ),

    "cfg_scale_15": CFGExperiment(
        name="CFG Scale 15.0",
        use_cfg=True,
        cfg_scale=15.0,
        cfg_dropout_rate=0.15,
        description="强guidance，可能过度拟合条件"
    ),
}


# ============================================================================
# 实验运行配置
# ============================================================================

EXPERIMENT_CONFIG = {
    # 训练配置（仅用于 use_cfg=True 的模型）
    "training": {
        "epochs": 500,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "use_fp16": True,
        "use_ema": True,
        "ema_decay": 0.9999,
    },

    # 推理配置（用于所有对比）
    "inference": {
        "ddim_steps": 40,
        "ddim_eta": 0.0,
    },

    # 评估样本数
    "eval_samples": 100,
}


def get_cfg_scales_to_compare() -> List[float]:
    """获取要对比的CFG scale列表"""
    return [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]


def list_experiments() -> None:
    """打印所有可用实验"""
    print("=" * 60)
    print("CFG 对比实验")
    print("=" * 60)
    for key, exp in CFG_EXPERIMENTS.items():
        print(f"\n[{key}] {exp.name}")
        print(f"    CFG: {'启用' if exp.use_cfg else '禁用'}")
        print(f"    Scale: {exp.cfg_scale}")
        print(f"    说明: {exp.description}")
