param(
    [Parameter(Position=0)][string]$Command = 'summary',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HOME = 'Q:\NAVA-Workspace\Caches\HuggingFace'
$python = $env:DNB_PYTHON
if (-not $python) {
    $python = 'Q:\NAVA-Workspace\Caches\VirtualEnvs\rhythm-dnb-qwen35-gpu\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Set DNB_PYTHON to a Python environment with requirements.txt installed.'
}
& $python (Join-Path $PSScriptRoot 'run.py') $Command @Arguments
exit $LASTEXITCODE
