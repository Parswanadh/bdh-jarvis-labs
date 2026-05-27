# PowerShell Script to run Local Logit Generation (Node 0)

Write-Host "======================================="
Write-Host "HYDRA: STARTING LOCAL NODE 0"
Write-Host "======================================="
Write-Host "This will process the first half of the TinyStories dataset."
Write-Host "This process will take several hours. You can stop it with Ctrl+C."
Write-Host ""

# Activate conda environment and run the script
conda activate bdh-fest
python generate_logits.py --node 0 --total-nodes 2

Write-Host "======================================="
Write-Host "LOCAL NODE 0 COMPLETE"
Write-Host "======================================="
Read-Host -Prompt "Press Enter to exit"
