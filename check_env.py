"""
Environment verification script for Chess GRPO Trainer.
This script checks that all required packages are installed and available.
"""

import importlib
import sys
import os

def check_package(package_name):
    """Check if a package is installed and report its version."""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        elif hasattr(module, 'version'):
            version = module.version
        else:
            version = "Unknown version"
        print(f"✅ {package_name} - {version}")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False

def check_stockfish():
    """Check if Stockfish is available."""
    try:
        from stockfish import Stockfish
        
        # Try with explicit path first (the one added to chess_grpo.py)
        stockfish_path = "/usr/games/stockfish"
        
        if os.path.exists(stockfish_path):
            try:
                stockfish = Stockfish(path=stockfish_path)
                print(f"✅ Stockfish engine - Available at {stockfish_path}")
                return True
            except Exception as e:
                print(f"❌ Stockfish engine - Error initializing with path {stockfish_path}: {e}")
        
        # Try default path as fallback
        try:
            stockfish = Stockfish()
            print(f"✅ Stockfish engine - Available (default path)")
            return True
        except Exception as e:
            print(f"❌ Stockfish engine - Not available: {e}")
            print("   Note: Stockfish is optional but recommended for full functionality.")
            return False
    except ImportError:
        print("❌ stockfish Python package - Not installed")
        return False

def check_gpu():
    """Check if GPU with CUDA is available."""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            
            print(f"✅ CUDA - Available with {gpu_count} GPU(s):")
            for i, name in enumerate(gpu_names):
                print(f"   - GPU {i}: {name}")
            
            if gpu_count >= 2:
                rtx_4090_count = sum(1 for name in gpu_names if "4090" in name)
                if rtx_4090_count >= 2:
                    print(f"✅ Dual RTX 4090s detected - Optimized for this configuration!")
                else:
                    print(f"ℹ️ Found {gpu_count} GPUs, but not dual RTX 4090s - Script will adapt automatically")
            return True
        else:
            print("❌ CUDA - Not available")
            return False
    except ImportError:
        print("❌ PyTorch - Not installed")
        return False

def main():
    """Check all required dependencies."""
    print("Checking Chess GRPO Trainer environment...\n")
    
    # Check Python version
    py_version = sys.version.split()[0]
    print(f"Python version: {py_version}")
    
    # Check required packages
    packages = [
        "torch", 
        "chess",
        "datasets",
        "transformers",
        "peft",
        "trl",
        "stockfish",
        "wandb",
        "accelerate",
        "deepspeed",
        "ninja",
        "psutil"
    ]
    
    print("\nChecking required packages:")
    results = [check_package(pkg) for pkg in packages]
    
    print("\nChecking Stockfish engine:")
    stockfish_ok = check_stockfish()
    
    print("\nChecking GPU support:")
    gpu_ok = check_gpu()
    
    # Summary
    print("\nEnvironment check summary:")
    if all(results) and stockfish_ok and gpu_ok:
        print("✅ All required packages and dependencies are installed.")
        print("✅ GPU support is available for optimal training performance.")
        print("✅ Environment is ready for training the Chess GRPO model.")
    elif all(results) and not stockfish_ok and gpu_ok:
        print("⚠️ All required packages are installed, but Stockfish engine is not available.")
        print("⚠️ You can still run the code in ablation mode, but full rewards won't work without Stockfish.")
    elif all(results) and stockfish_ok and not gpu_ok:
        print("⚠️ All required packages and Stockfish are installed, but GPU support is not available.")
        print("⚠️ Training will be extremely slow without GPU acceleration.")
    else:
        print("❌ Some requirements are missing. Please address the issues above.")
        print("   For packages, run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()