#!/bin/bash
# Chess GRPO RunPod Setup Script
# Usage: bash setup_runpod.sh [--repo REPO_URL] [--wandb-key API_KEY] [--hf-token HF_TOKEN] [--hub-repo HF_REPO]

set -e  # Exit on any error

# Default values
REPO_URL="https://github.com/darrenangle/grpo-chess.git"
WANDB_API_KEY=""
HF_TOKEN=""
HF_REPO=""
BRANCH="main"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo)
      REPO_URL="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --wandb-key)
      WANDB_API_KEY="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --hub-repo)
      HF_REPO="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "======================================="
echo "GRPO Chess RunPod Setup Script"
echo "======================================="
echo "This script will set up the GRPO Chess training environment"
echo "on a RunPod H200 instance and start training."
echo

# Check if we're running on a RunPod H200 instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script should be run on a GPU instance."
    exit 1
fi

# Check GPU type
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "Detected GPU: $GPU_NAME"
if [[ ! "$GPU_NAME" == *"H200"* ]]; then
    echo "Warning: This script is optimized for NVIDIA H200 GPUs."
    echo "Running on another GPU type may require configuration changes."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Checking system dependencies..."

# Installing system dependencies
echo "Installing system packages..."
apt-get update
apt-get install -y git python3-venv python3-pip stockfish tmux htop

# Clone the repository
echo "Cloning repository from $REPO_URL (branch: $BRANCH)..."
git clone --branch $BRANCH $REPO_URL /tmp/grpo-chess
cd /tmp/grpo-chess

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
if [ -f check_env.py ]; then
    python check_env.py
else
    echo "Warning: check_env.py not found, skipping environment verification."
fi

# Setup authentication for Weights & Biases
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Setting up Weights & Biases authentication..."
    echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc
    export WANDB_API_KEY=$WANDB_API_KEY
    
    # Try to verify the API key
    echo "Verifying Weights & Biases API key..."
    if pip install wandb > /dev/null && python -c "import wandb; wandb.login(key='$WANDB_API_KEY')" 2>/dev/null; then
        echo "✅ Weights & Biases authentication successful!"
    else
        echo "⚠️ Warning: Could not verify Weights & Biases API key. Check your key and try again."
    fi
else
    echo "No Weights & Biases API key provided. Training will proceed without WandB logging."
    echo "To add WandB logging later, run: wandb login"
fi

# Setup authentication for Hugging Face Hub
if [ ! -z "$HF_TOKEN" ]; then
    echo "Setting up Hugging Face Hub authentication..."
    echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
    export HF_TOKEN=$HF_TOKEN
    
    # Try to verify the token
    echo "Verifying Hugging Face Hub token..."
    if pip install huggingface_hub > /dev/null && python -c "from huggingface_hub import HfApi; api = HfApi(); api.whoami(token='$HF_TOKEN')" 2>/dev/null; then
        echo "✅ Hugging Face Hub authentication successful!"
    else
        echo "⚠️ Warning: Could not verify Hugging Face Hub token. Check your token and try again."
    fi
else
    echo "No Hugging Face Hub token provided. Models will only be saved locally."
    echo "To add HF Hub integration later, run: huggingface-cli login"
fi

# Move to home directory
mkdir -p ~/grpo-chess
cp -r /tmp/grpo-chess/* ~/grpo-chess/
cd ~/grpo-chess

# Create a tmux session for training
echo "Creating tmux session for training..."
TMUX_SESSION_NAME="grpo-training"

# Kill the session if it already exists
tmux kill-session -t $TMUX_SESSION_NAME 2>/dev/null || true

# Create a new session
tmux new-session -d -s $TMUX_SESSION_NAME

# Start training in the tmux session
if [ ! -z "$HF_REPO" ]; then
    echo "Starting training with Hugging Face Hub integration..."
    tmux send-keys -t $TMUX_SESSION_NAME "cd ~/grpo-chess && source venv/bin/activate && python chess_grpo.py --push-to-hub $HF_REPO --direct" C-m
else
    echo "Starting training without Hugging Face Hub integration..."
    tmux send-keys -t $TMUX_SESSION_NAME "cd ~/grpo-chess && source venv/bin/activate && python chess_grpo.py --direct" C-m
fi

echo
echo "======================================="
echo "Setup complete! 🎉"
echo "======================================="
echo
echo "Training is running in a tmux session. To attach to it:"
echo "  tmux attach-session -t $TMUX_SESSION_NAME"
echo
echo "To detach from the session (leave it running):"
echo "  Press Ctrl+B, then D"
echo
echo "Training logs will be visible in the tmux session."
echo "Models are being saved to:"
echo "  Local: ~/grpo-chess/outputs/"
if [ ! -z "$HF_REPO" ]; then
    echo "  Hugging Face Hub: https://huggingface.co/$HF_REPO"
fi
echo
echo "Monitor GPU usage with:"
echo "  watch -n 1 nvidia-smi"
echo
echo "Happy training! 🤖♟️"