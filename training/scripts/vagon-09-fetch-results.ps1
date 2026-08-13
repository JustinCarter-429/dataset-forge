param([string]$Root='C:\DatasetForgeCritic')
Get-ChildItem (Join-Path $Root 'reports'),(Join-Path $Root 'results') -Recurse -File -ErrorAction SilentlyContinue | Select-Object FullName,Length
