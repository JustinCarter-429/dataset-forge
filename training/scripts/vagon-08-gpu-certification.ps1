param([string]$Root='C:\DatasetForgeCritic')
& (Join-Path $Root 'venv\Scripts\python.exe') -c "import torch; assert torch.cuda.is_available(), 'GPU_CERTIFICATION_BLOCKED: CUDA unavailable'; print(torch.cuda.get_device_name(0))"
Write-Host 'Run the certified WP5A GPU command only after CUDA preflight succeeds; never start production training here.'
