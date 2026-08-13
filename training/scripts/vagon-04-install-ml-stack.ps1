param([string]$Root='C:\DatasetForgeCritic')
& (Join-Path $Root 'venv\Scripts\python.exe') -m pip install -r (Join-Path $Root 'handoff\dependencies\gemma4-qlora.txt')
