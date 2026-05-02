"""
WandB 实验管理工具

提供统一的 wandb 日志记录接口，包括：
- 训练/验证损失曲线
- 过拟合分析（train-val loss 差异）
- 样本可视化
- 模型性能指标
"""
import os
import numpy as np
import torch
from datetime import datetime

# 设置 WandB API Key（在导入 wandb 前设置）
WANDB_API_KEY = "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"
if "WANDB_API_KEY" not in os.environ:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb not installed, logging disabled")


class WandbLogger:
    """WandB 日志管理器"""

    def __init__(
        self,
        project: str = "cbct2sct_IBA",
        name: str = None,
        config: dict = None,
        tags: list = None,
        notes: str = None,
        enabled: bool = True,
    ):
        """
        初始化 WandB Logger

        Args:
            project: wandb 项目名
            name: 实验名称（默认自动生成）
            config: 实验配置字典
            tags: 标签列表
            notes: 实验备注
            enabled: 是否启用 wandb
        """
        self.enabled = enabled and HAS_WANDB
        self.run = None

        if not self.enabled:
            print("📊 WandB logging disabled")
            return

        # 生成默认实验名
        if name is None:
            name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            reinit=True,
        )

        print(f"📊 WandB initialized: {self.run.url}")

    def log_config(self, config: dict):
        """记录配置"""
        if not self.enabled:
            return
        wandb.config.update(config)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        记录指标

        Args:
            metrics: 指标字典，如 {'train_loss': 0.1, 'val_loss': 0.12}
            step: 步数（epoch）
        """
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_training_step(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float = None,
        extra_metrics: dict = None,
    ):
        """
        记录训练步骤

        自动计算并记录：
        - train_loss, val_loss
        - loss_gap (train - val，用于检测过拟合)
        - loss_ratio (val / train)

        Args:
            epoch: 当前 epoch
            train_loss: 训练损失
            val_loss: 验证损失
            learning_rate: 当前学习率
            extra_metrics: 额外指标
        """
        if not self.enabled:
            return

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_gap": val_loss - train_loss,  # 正值表示可能过拟合
            "loss_ratio": val_loss / train_loss if train_loss > 0 else 0,
        }

        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        if extra_metrics:
            metrics.update(extra_metrics)

        wandb.log(metrics, step=epoch)

    def log_image(
        self,
        name: str,
        image: np.ndarray,
        caption: str = None,
        step: int = None,
    ):
        """
        记录图像

        Args:
            name: 图像名称
            image: numpy 数组 (H, W) 或 (H, W, C)
            caption: 图像说明
            step: 步数
        """
        if not self.enabled:
            return

        wandb.log({
            name: wandb.Image(image, caption=caption)
        }, step=step)

    def log_comparison_images(
        self,
        cbct: np.ndarray,
        ct_gt: np.ndarray,
        sct_pred: np.ndarray,
        step: int = None,
        sample_idx: int = 0,
    ):
        """
        记录 CBCT/CT/sCT 对比图像

        Args:
            cbct: CBCT 输入
            ct_gt: CT ground truth
            sct_pred: 预测的 sCT
            step: 步数
            sample_idx: 样本索引
        """
        if not self.enabled:
            return

        # 确保是 2D 图像
        if len(cbct.shape) > 2:
            cbct = cbct.squeeze()
        if len(ct_gt.shape) > 2:
            ct_gt = ct_gt.squeeze()
        if len(sct_pred.shape) > 2:
            sct_pred = sct_pred.squeeze()

        # 计算差异图
        diff = np.abs(ct_gt - sct_pred)

        wandb.log({
            f"samples/cbct_{sample_idx}": wandb.Image(
                cbct, caption="CBCT Input"
            ),
            f"samples/ct_gt_{sample_idx}": wandb.Image(
                ct_gt, caption="CT Ground Truth"
            ),
            f"samples/sct_pred_{sample_idx}": wandb.Image(
                sct_pred, caption="sCT Prediction"
            ),
            f"samples/diff_{sample_idx}": wandb.Image(
                diff, caption=f"Difference (MAE={diff.mean():.2f})"
            ),
        }, step=step)

    def log_evaluation_metrics(
        self,
        mae: float,
        psnr: float,
        ssim: float,
        rmse: float = None,
        step: int = None,
        prefix: str = "eval",
    ):
        """
        记录评估指标

        Args:
            mae: Mean Absolute Error (HU)
            psnr: Peak Signal-to-Noise Ratio (dB)
            ssim: Structural Similarity Index
            rmse: Root Mean Square Error (HU)
            step: 步数
            prefix: 指标前缀
        """
        if not self.enabled:
            return

        metrics = {
            f"{prefix}/MAE": mae,
            f"{prefix}/PSNR": psnr,
            f"{prefix}/SSIM": ssim,
        }

        if rmse is not None:
            metrics[f"{prefix}/RMSE"] = rmse

        wandb.log(metrics, step=step)

    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: int = None,
    ):
        """记录直方图"""
        if not self.enabled:
            return

        wandb.log({
            name: wandb.Histogram(values)
        }, step=step)

    def log_model_artifact(
        self,
        model_path: str,
        name: str = "model",
        metadata: dict = None,
    ):
        """
        保存模型为 artifact

        Args:
            model_path: 模型文件路径
            name: artifact 名称
            metadata: 元数据
        """
        if not self.enabled:
            return

        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def create_loss_table(self, train_losses: list, val_losses: list):
        """
        创建损失表格（用于后续分析）

        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
        """
        if not self.enabled:
            return

        table = wandb.Table(
            columns=["epoch", "train_loss", "val_loss", "gap"],
            data=[
                [i, t, v, v - t]
                for i, (t, v) in enumerate(zip(train_losses, val_losses))
            ]
        )
        wandb.log({"loss_table": table})

    def finish(self):
        """结束记录"""
        if self.enabled and self.run:
            wandb.finish()
            print("📊 WandB run finished")


def setup_wandb(
    experiment_name: str,
    architecture: str,
    config: dict,
    tags: list = None,
) -> WandbLogger:
    """
    快速设置 wandb

    Args:
        experiment_name: 实验名称
        architecture: UNet 架构名称
        config: 训练配置
        tags: 标签列表

    Returns:
        WandbLogger 实例
    """
    full_tags = [architecture]
    if tags:
        full_tags.extend(tags)

    return WandbLogger(
        project="cbct2sct_IBA",
        name=f"{experiment_name}_{architecture}",
        config={
            "architecture": architecture,
            **config,
        },
        tags=full_tags,
        enabled=True,
    )
