#!/bin/bash
#
# 架构对比完整工作流
# 包括: 推理 → 评估 → 生成报告
#
# 用法:
#   bash scripts/compare_architectures.sh
#   bash scripts/compare_architectures.sh --num-samples 100  # 快速测试
#

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数 (匹配训练配置)
NUM_SAMPLES=508  # 默认全部测试集
DDIM_STEPS=40    # 训练时用40步，推理保持一致
BATCH_SIZE=16

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --ddim-steps)
            DDIM_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}架构对比完整工作流${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "测试样本数: ${NUM_SAMPLES}"
echo -e "DDIM步数: ${DDIM_STEPS}"
echo -e "批次大小: ${BATCH_SIZE}"
echo -e "${BLUE}======================================================================${NC}\n"

# 定义架构列表
ARCHITECTURES=("concatenation" "skip" "cross_attention")

# 定义模型路径映射
declare -A ARCH_DIRS
ARCH_DIRS["concatenation"]="exp_concatenation_noCFG"
ARCH_DIRS["skip"]="skip_noCFG_20260120_154721_ep200"
ARCH_DIRS["cross_attention"]="cross_attention_noCFG_20260121_103603_ep200"

# 创建输出目录
OUTPUT_BASE="outputs/architecture_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_BASE}"

echo -e "${GREEN}输出目录: ${OUTPUT_BASE}${NC}\n"

# ============================================================================
# 阶段1: 推理 - 生成所有架构的sCT图像
# ============================================================================

echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}阶段1: 推理生成sCT图像${NC}"
echo -e "${YELLOW}======================================================================${NC}\n"

for arch in "${ARCHITECTURES[@]}"; do
    echo -e "${GREEN}>>> 推理: ${arch}${NC}"

    # 获取模型目录
    MODEL_DIR="checkpoints/${ARCH_DIRS[$arch]}"

    # 自动查找best模型（优先级：unet_best.pth > unet_best_ep*.pth > unet.pth）
    if [ -f "${MODEL_DIR}/unet_best.pth" ]; then
        UNET_PATH="${MODEL_DIR}/unet_best.pth"
        echo -e "${YELLOW}使用 unet_best.pth${NC}"
    elif [ -n "$(ls ${MODEL_DIR}/unet_best_ep*.pth 2>/dev/null)" ]; then
        # 找到最新的best模型（按修改时间排序）
        UNET_PATH=$(ls -t ${MODEL_DIR}/unet_best_ep*.pth 2>/dev/null | head -1)
        echo -e "${YELLOW}使用 $(basename ${UNET_PATH})${NC}"
    elif [ -f "${MODEL_DIR}/unet.pth" ]; then
        UNET_PATH="${MODEL_DIR}/unet.pth"
        echo -e "${YELLOW}⚠️  使用 unet.pth${NC}"
    else
        echo -e "${RED}❌ 未找到模型文件${NC}"
        continue
    fi

    # 查找VAE路径 (优先使用模型目录中的VAE，否则使用共享VAE)
    if [ -f "${MODEL_DIR}/vae.pth" ]; then
        VAE_PATH="${MODEL_DIR}/vae.pth"
    elif [ -f "checkpoints/exp_concatenation_noCFG/vae.pth" ]; then
        VAE_PATH="checkpoints/exp_concatenation_noCFG/vae.pth"
    else
        echo -e "${RED}❌ 未找到VAE模型${NC}"
        continue
    fi

    # 创建输出目录
    INFERENCE_OUTPUT="${OUTPUT_BASE}/${arch}/inference_output"
    mkdir -p "${INFERENCE_OUTPUT}"

    # 推理命令 (匹配训练配置: 40步DDIM + 纯随机初始化)
    echo "  模型: ${UNET_PATH}"
    echo "  VAE: ${VAE_PATH}"
    echo "  输出: ${INFERENCE_OUTPUT}"

    python scripts/infer.py \
        --vae "${VAE_PATH}" \
        --unet "${UNET_PATH}" \
        --input data/dataset/manifest.csv \
        --output "${INFERENCE_OUTPUT}" \
        --ddim-steps ${DDIM_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --save-npy \
        --save-vis \
        --vis-freq 10 \
        2>&1 | tee "${OUTPUT_BASE}/${arch}_inference.log"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${arch} 推理完成${NC}\n"
    else
        echo -e "${RED}✗ ${arch} 推理失败${NC}\n"
    fi
done

# ============================================================================
# 阶段2: 评估 - 计算每个架构的指标
# ============================================================================

echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}阶段2: 评估生成质量${NC}"
echo -e "${YELLOW}======================================================================${NC}\n"

for arch in "${ARCHITECTURES[@]}"; do
    echo -e "${GREEN}>>> 评估: ${arch}${NC}"

    INFERENCE_OUTPUT="${OUTPUT_BASE}/${arch}/inference_output"
    EVALUATION_OUTPUT="${OUTPUT_BASE}/${arch}/evaluation_results"

    # 检查推理输出是否存在
    if [ ! -d "${INFERENCE_OUTPUT}" ]; then
        echo -e "${RED}❌ 未找到推理输出: ${INFERENCE_OUTPUT}${NC}"
        continue
    fi

    mkdir -p "${EVALUATION_OUTPUT}"

    # 评估命令 (10月版本: full evaluation with HU ranges + visualizations + markdown report)
    python scripts/eval.py \
        --pred "${INFERENCE_OUTPUT}" \
        --gt data/dataset/CT \
        --output "${EVALUATION_OUTPUT}" \
        --full \
        --data-range 2000.0 \
        2>&1 | tee "${OUTPUT_BASE}/${arch}_evaluation.log"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${arch} 评估完成${NC}\n"
    else
        echo -e "${RED}✗ ${arch} 评估失败${NC}\n"
    fi
done

# ============================================================================
# 阶段3: 生成对比报告
# ============================================================================

echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}阶段3: 生成对比报告${NC}"
echo -e "${YELLOW}======================================================================${NC}\n"

# 创建汇总报告
SUMMARY_REPORT="${OUTPUT_BASE}/architecture_comparison_summary.md"

cat > "${SUMMARY_REPORT}" << EOF
# 架构对比报告

**生成时间:** $(date '+%Y-%m-%d %H:%M:%S')
**测试样本数:** ${NUM_SAMPLES}
**DDIM步数:** ${DDIM_STEPS}

---

## 评估结果汇总

EOF

# 提取每个架构的关键指标
echo -e "${GREEN}汇总各架构结果...${NC}\n"

for arch in "${ARCHITECTURES[@]}"; do
    EVAL_JSON="${OUTPUT_BASE}/${arch}/evaluation_results/evaluation_results.json"

    if [ -f "${EVAL_JSON}" ]; then
        echo "### ${arch}" >> "${SUMMARY_REPORT}"
        echo "" >> "${SUMMARY_REPORT}"

        # 使用Python提取指标 (从10月版本的新格式)
        python3 << PYTHON_SCRIPT >> "${SUMMARY_REPORT}"
import json
import sys

try:
    with open('${EVAL_JSON}', 'r') as f:
        data = json.load(f)

    # 提取Full Range HU范围指标
    summary_hu = data.get('summary_hu_ranges', {})
    full_range = summary_hu.get('Full Range', {})

    if full_range:
        print("| Metric | Mean | Std |")
        print("|--------|------|-----|")
        print(f"| MAE (HU) | {full_range.get('MAE', {}).get('mean', 0):.2f} | {full_range.get('MAE', {}).get('std', 0):.2f} |")
        print(f"| RMSE (HU) | {full_range.get('RMSE', {}).get('mean', 0):.2f} | {full_range.get('RMSE', {}).get('std', 0):.2f} |")
        print(f"| PSNR (dB) | {full_range.get('PSNR', {}).get('mean', 0):.2f} | {full_range.get('PSNR', {}).get('std', 0):.2f} |")
        print(f"| SSIM | {full_range.get('SSIM', {}).get('mean', 0):.4f} | {full_range.get('SSIM', {}).get('std', 0):.4f} |")
    else:
        # 回退到整体summary
        summary = data.get('summary', {})
        print("| Metric | Mean | Std |")
        print("|--------|------|-----|")
        print(f"| MAE (HU) | {summary.get('MAE', {}).get('mean', 0):.2f} | {summary.get('MAE', {}).get('std', 0):.2f} |")
        print(f"| RMSE (HU) | {summary.get('RMSE', {}).get('mean', 0):.2f} | {summary.get('RMSE', {}).get('std', 0):.2f} |")
        print(f"| PSNR (dB) | {summary.get('PSNR', {}).get('mean', 0):.2f} | {summary.get('PSNR', {}).get('std', 0):.2f} |")
        print(f"| SSIM | {summary.get('SSIM', {}).get('mean', 0):.4f} | {summary.get('SSIM', {}).get('std', 0):.4f} |")
    print("")
except Exception as e:
    print(f"Error processing ${arch}: {e}", file=sys.stderr)
PYTHON_SCRIPT

    else
        echo "### ${arch}" >> "${SUMMARY_REPORT}"
        echo "" >> "${SUMMARY_REPORT}"
        echo "❌ 评估结果未找到" >> "${SUMMARY_REPORT}"
        echo "" >> "${SUMMARY_REPORT}"
    fi
done

# 添加结论部分
cat >> "${SUMMARY_REPORT}" << EOF

---

## 详细结果

各架构的详细评估结果保存在:
EOF

for arch in "${ARCHITECTURES[@]}"; do
    echo "- **${arch}**: \`${OUTPUT_BASE}/${arch}/evaluation_results/\`" >> "${SUMMARY_REPORT}"
done

cat >> "${SUMMARY_REPORT}" << EOF

## 可视化

各架构的可视化对比图保存在对应的 \`evaluation_results/visualizations/\` 目录下。

## 推理输出

生成的sCT NPY文件保存在对应的 \`inference_output/\` 目录下。

EOF

# ============================================================================
# 完成
# ============================================================================

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}✓ 架构对比工作流完成！${NC}"
echo -e "${GREEN}======================================================================${NC}\n"

echo -e "结果保存在: ${BLUE}${OUTPUT_BASE}${NC}\n"

echo -e "查看汇总报告:"
echo -e "  ${BLUE}cat ${SUMMARY_REPORT}${NC}\n"

echo -e "查看详细结果:"
for arch in "${ARCHITECTURES[@]}"; do
    echo -e "  ${arch}: ${BLUE}${OUTPUT_BASE}/${arch}/evaluation_results/evaluation_report.md${NC}"
done

echo ""
