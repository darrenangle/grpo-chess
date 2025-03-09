"""
Environment verification script for Chess GRPO Trainer.
This script checks that all required packages are installed and available.
"""

import importlib
import sys

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
        stockfish = Stockfish()
        print(f"✅ Stockfish engine - Available")
        return True
    except Exception as e:
        print(f"❌ Stockfish engine - Not available: {e}")
        print("   Note: Stockfish is optional but recommended for full functionality.")
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
        "wandb"
    ]
    
    print("\nChecking required packages:")
    results = [check_package(pkg) for pkg in packages]
    
    print("\nChecking Stockfish engine:")
    stockfish_ok = check_stockfish()
    
    # Summary
    print("\nEnvironment check summary:")
    if all(results) and stockfish_ok:
        print("✅ All required packages and dependencies are installed.")
        print("✅ Environment is ready for training the Chess GRPO model.")
    elif all(results) and not stockfish_ok:
        print("⚠️ All required packages are installed, but Stockfish engine is not available.")
        print("⚠️ You can still run the code in ablation mode, but full rewards won't work without Stockfish.")
    else:
        print("❌ Some required packages are missing. Please install them with:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()