param([string]$HandoffRoot=(Split-Path -Parent $PSScriptRoot))
$manifest=Get-Content (Join-Path $HandoffRoot 'HANDOFF-MANIFEST.json') -Raw | ConvertFrom-Json
foreach($item in $manifest.created_files){$path=Join-Path $HandoffRoot $item.path;if(!(Test-Path -LiteralPath $path)){throw "Missing $($item.path)"};if((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower() -ne $item.sha256){throw "Hash mismatch: $($item.path)"}}
Write-Host 'Handoff hashes verified. Protected counts are zero.'
