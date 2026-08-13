param([string]$Root='C:\DatasetForgeCritic')
$env:HF_HOME=Join-Path $Root 'hf-cache'
& (Join-Path $Root 'venv\Scripts\python.exe') -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('google/gemma-4-E4B-it',revision='ee0ef6023621cff504d758262d4e04895a5af4a2,local_files_only=True); assert type(t).__name__=='GemmaTokenizer' and len(t)==262144 and t.chat_template; print('Pinned Gemma tokenizer verified')"
