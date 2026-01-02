#!/bin/bash
# Setup script for 6-Max training environment
# Usage: bash scripts/setup_env.sh [dev|training|all]

set -e

MODE=${1:-"dev"}

echo "=========================================="
echo "6-Max Poker AI Environment Setup"
echo "Mode: $MODE"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "uv: $(uv --version) ✓"

# Sync dependencies based on mode
case $MODE in
    "dev")
        echo "Installing development dependencies..."
        uv sync --extra dev
        ;;
    "training")
        echo "Installing training dependencies..."
        uv sync --extra training
        ;;
    "all")
        echo "Installing all dependencies..."
        uv sync --extra all
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash scripts/setup_env.sh [dev|training|all]"
        exit 1
        ;;
esac

# Verify installation
echo ""
echo "Verifying installation..."
uv run python -c "import sixmax; print('sixmax package: ✓')"
uv run python -c "import torch; print(f'PyTorch: {torch.__version__} ✓')"

if [ "$MODE" = "training" ] || [ "$MODE" = "all" ]; then
    uv run python -c "import wandb; print('wandb: ✓')" 2>/dev/null || echo "wandb: not installed"
    echo ""
    echo "Wandb 配置提示:"
    echo "  1. 获取 API Key: https://wandb.ai/settings"
    echo "  2. 在 scripts/submit_slurm.sh 中设置 WANDB_API_KEY"
    echo "  或运行: wandb login"
fi

# Check PyTorch CUDA support (不需要实际 GPU)
uv run python -c "
import torch
import sys
if sys.platform == 'linux':
    if torch.version.cuda:
        print(f'PyTorch CUDA support: {torch.version.cuda} ✓')
    else:
        print('Warning: PyTorch without CUDA support!')
elif sys.platform == 'darwin':
    print('macOS: CPU/MPS version ✓')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To run tests:"
echo "  uv run pytest tests/"
echo ""
echo "To train (local):"
echo "  uv run scripts/train.py --config test"
echo ""
echo "To train (SLURM):"
echo "  sbatch scripts/submit_slurm.sh full"
echo "=========================================="
