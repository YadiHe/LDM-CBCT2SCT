"""
基础配置 - 所有配置的父类
提供统一的配置接口和默认值
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any

# ============================================================================
# 基础路径配置
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


@dataclass
class BaseConfig:
    """基础配置类"""
    # 设备
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 随机种子
    seed: int = 42

    def __post_init__(self):
        """初始化后处理"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                result[key] = str(value)
            else:
                result[key] = value
        return result


@dataclass
class DataConfig(BaseConfig):
    """数据配置"""
    # 数据路径
    manifest_path: str = ""
    data_dir: str = DEFAULT_DATA_DIR

    # 数据集划分
    train_size: Optional[int] = None  # None表示使用全部
    val_size: Optional[int] = None
    test_size: Optional[int] = None

    # 数据加载
    batch_size: int = 32
    num_workers: int = 4

    # 图像尺寸
    target_size: Tuple[int, int] = (256, 256)

    # 预处理方式
    preprocess: str = "linear"  # "linear" 或 "tanh"

    # 数据增强
    augmentation: Optional[Dict] = field(default_factory=lambda: {
        'degrees': (-0.5, 0.5),
        'translate': (0.05, 0.05),
        'scale': (0.95, 1.05),
        'shear': None,
    })


@dataclass
class ModelConfig(BaseConfig):
    """模型配置"""
    # VAE
    vae_latent_dim: int = 3
    vae_base_channels: int = 64

    # UNet
    unet_base_channels: int = 256
    unet_dropout_rate: float = 0.1

    # UNet 类型: "concatenation", "skip", "cross_attention", "control_paca"
    unet_type: str = "concatenation"

    # Diffusion
    diffusion_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # CFG (Classifier-Free Guidance)
    use_cfg: bool = False
    cfg_dropout_rate: float = 0.15
    cfg_scale: float = 7.5


@dataclass
class TrainingConfig(BaseConfig):
    """训练配置"""
    # 训练参数
    epochs: int = 1000
    learning_rate: float = 1e-4
    warmup_epochs: int = 0
    warmup_lr: float = 0.0

    # 早停
    early_stopping: int = 30
    patience: int = 10

    # 混合精度
    use_fp16: bool = True

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # 保存
    epochs_between_prediction: int = 20
    save_dir: str = "trained_models_256"

    # VAE 损失权重
    vae_perceptual_weight: float = 0.1
    vae_ssim_weight: float = 1.0
    vae_mse_weight: float = 1.0
    vae_kl_weight: float = 1e-6
    vae_l1_weight: float = 0.0


@dataclass
class InferenceConfig(BaseConfig):
    """推理配置"""
    # 模型路径
    vae_path: str = ""
    unet_path: str = ""

    # 采样策略: "ddpm", "ddim"
    sampling_method: str = "ddim"
    ddim_steps: int = 40
    ddim_eta: float = 0.0

    # 时间步调度: "linear", "quadratic", "power"
    timestep_schedule: str = "linear"

    # CFG
    use_cfg: bool = False
    cfg_scale: float = 7.5

    # 输出
    output_dir: str = "outputs/inference"
    save_npy: bool = True
    save_nifti: bool = False


@dataclass
class EvaluationConfig(BaseConfig):
    """评估配置"""
    # 输入路径
    prediction_dir: str = ""
    ground_truth_dir: str = ""

    # 输出路径
    output_dir: str = "outputs/evaluation"

    # 评估选项
    compute_regional: bool = True  # 分区域评估
    save_visualizations: bool = True

    # HU 范围分区
    hu_regions: Dict = field(default_factory=lambda: {
        'full': (-1000, 1000),
        'soft_tissue': (-150, 150),
        'low_density': (-1000, -150),
        'high_density': (150, 1000),
    })


# ============================================================================
# 预设配置
# ============================================================================

def get_config_256() -> Dict[str, BaseConfig]:
    """获取 256×256 预设配置"""
    return {
        'data': DataConfig(
            target_size=(256, 256),
            batch_size=32,
        ),
        'model': ModelConfig(
            unet_base_channels=256,
        ),
        'training': TrainingConfig(
            save_dir="trained_models_256",
        ),
    }


def get_config_512() -> Dict[str, BaseConfig]:
    """获取 512×512 预设配置"""
    return {
        'data': DataConfig(
            target_size=(512, 512),
            batch_size=8,  # 512需要更小的batch
        ),
        'model': ModelConfig(
            unet_base_channels=128,  # 512可能需要减少通道
        ),
        'training': TrainingConfig(
            save_dir="trained_models_512",
        ),
    }


def get_config_cfg() -> Dict[str, BaseConfig]:
    """获取 CFG (Classifier-Free Guidance) 预设配置"""
    return {
        'data': DataConfig(
            target_size=(256, 256),
            batch_size=32,
        ),
        'model': ModelConfig(
            unet_base_channels=256,
            use_cfg=True,
            cfg_dropout_rate=0.15,
        ),
        'training': TrainingConfig(
            save_dir="trained_models_256Guidance",
            use_fp16=True,
            use_ema=True,
        ),
    }
