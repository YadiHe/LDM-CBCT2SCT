#!/usr/bin/env python
"""
WandB 配置脚本

用法:
    python setup_wandb.py

或者直接在终端设置环境变量:
    export WANDB_API_KEY="your_api_key"
"""
import os

# WandB API Key（v1 token 格式）
WANDB_API_KEY = "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"

def setup():
    """设置 WandB 环境变量"""
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    print("✓ WANDB_API_KEY 已设置")

    # 测试连接
    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        print("✓ WandB 登录成功")
        return True
    except Exception as e:
        print(f"⚠️  WandB 登录失败: {e}")
        print("请检查 API Key 是否正确")
        return False


if __name__ == "__main__":
    setup()
