#!/bin/bash
#SBATCH --partition=V100q
#SBATCH --nodelist=node22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --job-name=sixmax
#SBATCH --output=logs/6max_%j.out
#SBATCH --error=logs/6max_%j.err

# 6-Max Training SLURM Script
# Usage: sbatch scripts/submit_slurm.sh [config] [extra_args]
# Example: sbatch scripts/submit_slurm.sh full --wandb --run-name "exp1"

set -e

# ============ 手动配置区 ============
# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# Wandb 配置
# API Key: 从 https://wandb.ai/settings 获取
export WANDB_API_KEY="your-api-key-here"
# 离线模式 (服务器没网时启用，训练完后用 wandb sync 上传)
# export WANDB_MODE="offline"
# ====================================

# Configuration
CONFIG=${1:-"full"}
shift 2>/dev/null || true
EXTRA_ARGS="$@"

# Create logs directory
mkdir -p logs

# Print job info
echo "=========================================="
echo "6-Max Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="
echo ""

# Load modules (adjust for your cluster)
# module load python/3.11
# module load cuda/12.1

# Determine runner (prefer uv if available)
if command -v uv &> /dev/null; then
    RUNNER="uv run"
    echo "Using uv"
else
    # Fallback to venv
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi
    RUNNER="python"
    echo "Using python directly"
fi

# Check GPU availability
$RUNNER python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Run training
$RUNNER scripts/train.py \
    --config "$CONFIG" \
    --device cuda \
    --checkpoint-dir "checkpoints/$SLURM_JOB_ID" \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
