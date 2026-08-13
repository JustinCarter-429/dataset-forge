param([string]$Root='C:\DatasetForgeCritic')
python -m venv (Join-Path $Root 'venv')
& (Join-Path $Root 'venv\Scripts\python.exe') -m pip install --upgrade pip
