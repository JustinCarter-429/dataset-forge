$ErrorActionPreference = "Stop"

$TrainingRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$LogPath = Join-Path $PSScriptRoot "artifacts\critic-evaluation-results\logs\local-test-results.txt"
$ExitPath = Join-Path $PSScriptRoot "artifacts\critic-evaluation-results\logs\local-test-exit-code.txt"
$CompletionPath = Join-Path $PSScriptRoot "artifacts\critic-evaluation-results\logs\local-test-complete.txt"
$BaseTemp = Join-Path $TrainingRoot ".pytest-e1-verification-durable-final-v2"

if (Test-Path -LiteralPath $BaseTemp) {
    throw "Fresh pytest base temp path already exists: $BaseTemp"
}

New-Item -ItemType Directory -Path $BaseTemp | Out-Null
$FocusedTemp = Join-Path $BaseTemp "focused"
$RelevantTemp = Join-Path $BaseTemp "relevant"
$env:PYTHONPATH = Join-Path $TrainingRoot "src"
Set-Location -LiteralPath $TrainingRoot

$started = [DateTime]::UtcNow
@"
Phase E1 durable local verification tests
Started UTC: $($started.ToString("o"))
PYTHONPATH: training/src
Repository-owned base temp: training/.pytest-e1-verification-durable-final-v2
Optional exclusion: tests/test_live_dashboard.py
Exclusion reason: optional TensorBoard dependency is unavailable in the local test environment

"@ | Out-File -FilePath $LogPath -Encoding utf8

$combinedExit = 1
$focusedExit = 1
$relevantExit = 1
try {
    # Native test processes may intentionally write diagnostics to stderr. Preserve
    # those lines in the tee without making PowerShell terminate the pipeline.
    $ErrorActionPreference = "Continue"
    "=== Focused Phase E1 tests ===" | Tee-Object -FilePath $LogPath -Append
    & python -m pytest tests/test_e1_pipeline.py -o addopts= -ra "--basetemp=$FocusedTemp" 2>&1 |
        Tee-Object -FilePath $LogPath -Append
    $focusedExit = $LASTEXITCODE
    "Focused exit code: $focusedExit`n" | Tee-Object -FilePath $LogPath -Append

    "=== Relevant training suite (excluding only optional live-dashboard module) ===" |
        Tee-Object -FilePath $LogPath -Append
    & python -m pytest --ignore=tests/test_live_dashboard.py -o addopts= -ra "--basetemp=$RelevantTemp" 2>&1 |
        Tee-Object -FilePath $LogPath -Append
    $relevantExit = $LASTEXITCODE
    "Relevant-suite exit code: $relevantExit" | Tee-Object -FilePath $LogPath -Append

    $combinedExit = if ($focusedExit -eq 0 -and $relevantExit -eq 0) { 0 } else { 1 }
}
finally {
    $finished = [DateTime]::UtcNow
    "Finished UTC: $($finished.ToString("o"))" | Tee-Object -FilePath $LogPath -Append
    "Wall duration seconds: $(($finished - $started).TotalSeconds)" | Tee-Object -FilePath $LogPath -Append
    "$combinedExit" | Out-File -FilePath $ExitPath -Encoding ascii
    $completion = if ($combinedExit -eq 0) { "complete" } else { "failed" }
    "$completion" | Out-File -FilePath $CompletionPath -Encoding ascii
}

exit $combinedExit
