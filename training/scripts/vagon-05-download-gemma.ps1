param([string]$Root='C:\DatasetForgeCritic')
$env:HF_HOME=Join-Path $Root 'hf-cache'
& (Join-Path $Root 'venv\Scripts\python.exe') -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-4-E4B-it', revision='ee0ef6023621cff504d758262d4e04895a5af4a2'))"
