import subprocess
import sys
import time
import os
from pathlib import Path

def run_node():
    # Node 0 is your local laptop
    cmd = [sys.executable, "generate_logits.py", "--node", "0", "--total-nodes", "2"]
    
    print("="*60)
    print("HYDRA LOGIT GENERATOR: LOCAL NODE 0")
    print("="*60)
    print("Status: Thermal-Guarded & Self-Healing Active")
    
    while True:
        try:
            # Run the generator
            process = subprocess.Popen(cmd)
            process.wait()
            
            if process.returncode == 0:
                print("\n[SUCCESS] Local Node 0 has finished its half of the dataset!")
                break
            else:
                print(f"\n[CRASH] Generator exited with code {process.returncode}")
                print("Restarting in 10 seconds...")
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n[STOPPED] User interrupted. Saving progress and exiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            time.sleep(10)

if __name__ == "__main__":
    # Ensure the environment is set up
    if not Path("generate_logits.py").exists():
        print("Error: generate_logits.py not found in current directory.")
        sys.exit(1)
        
    run_node()
