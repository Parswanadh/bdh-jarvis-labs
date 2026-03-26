import subprocess
import sys
import os

def install_requirements():
    print("="*60)
    print("HYDRA ENVIRONMENT SETUP")
    print("="*60)
    
    packages = ["torch", "transformers", "accelerate", "psutil", "huggingface_hub"]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
    print("\n[SUCCESS] Environment is ready for logit generation.")

if __name__ == "__main__":
    install_requirements()
