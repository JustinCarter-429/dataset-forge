param([string]$Root='C:\DatasetForgeCritic',[string]$HandoffRoot=(Split-Path -Parent $PSScriptRoot))
@('handoff','workspace','hf-cache','checkpoints','rendered','reports','logs','results')|ForEach-Object{New-Item -ItemType Directory -Force -Path (Join-Path $Root $_)|Out-Null}
Copy-Item -LiteralPath $HandoffRoot -Destination (Join-Path $Root 'handoff') -Recurse -Force
Write-Host "Workspace prepared at $Root."
