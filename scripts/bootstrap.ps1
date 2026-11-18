$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$VenvName = if ($env:HELIOS_VENV) { $env:HELIOS_VENV } elseif (Test-Path (Join-Path $Root 'venv')) { 'venv' } elseif (Test-Path (Join-Path $Root '.venv')) { '.venv' } else { 'venv' }
$VenvPath = Join-Path $Root $VenvName
$PythonExe = 'python'

Set-Location $Root
Write-Host '[Helios] Bootstrapping environment...'

if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonExe = 'py -3'
    } else {
        throw '[Helios] Python 3 is required but was not found in PATH.'
    }
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "[Helios] Creating virtual environment at $VenvPath"
    Invoke-Expression "$PythonExe -m venv `"$VenvPath`""
}

$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    throw '[Helios] Could not locate venv python executable.'
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r requirements.txt
& $VenvPython -m pip install -e .

if (Test-Path (Join-Path $Root 'frontend\package.json')) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host '[Helios] Installing frontend dependencies...'
        Push-Location (Join-Path $Root 'frontend')
        npm install
        Pop-Location
    } else {
        Write-Warning '[Helios] npm not found. Install Node.js 18+ and run: cd frontend; npm install'
    }
}

Write-Host '[Helios] Bootstrap completed successfully.'
