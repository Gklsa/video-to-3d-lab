param(
    [string]$Python = '',
    [int]$Port = 8871,
    [string]$Clip = ''
)
$ErrorActionPreference = 'Stop'
$labPython = $Python
if (-not $labPython) {
    $labCandidates = @(
        (Join-Path $PSScriptRoot 'butterfly-mose3\.venv\Scripts\python.exe'),
        (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
    )
    $labPython = $labCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $labPython) { $labPython = 'python' }
}
if (-not (Get-Command $labPython -ErrorAction SilentlyContinue)) {
    throw "Python not found: $labPython. Activate your environment or pass -Python <path>."
}
$labLaunchArguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'start_butterfly.py'), '--port', $Port)
if ($Clip) { $labLaunchArguments += @('--clip', $Clip) }
& $labPython @labLaunchArguments
if ($LASTEXITCODE -ne 0) { throw "Viewer exited with code $LASTEXITCODE" }
