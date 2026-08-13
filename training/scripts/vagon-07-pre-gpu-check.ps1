param([string]$Root='C:\DatasetForgeCritic')
& (Join-Path $Root 'venv\Scripts\python.exe') -c "import torch; print({'cuda':torch.cuda.is_available(),'torch':torch.__version__,'gpu_count':torch.cuda.device_count()})"
Write-Host '=== STOP CHEAP TIER HERE ==='
