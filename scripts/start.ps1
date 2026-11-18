$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$VenvName = if ($env:HELIOS_VENV) { $env:HELIOS_VENV } elseif (Test-Path (Join-Path $Root 'venv')) { 'venv' } elseif (Test-Path (Join-Path $Root '.venv')) { '.venv' } else { 'venv' }
$VenvPython = Join-Path $Root ("$VenvName\Scripts\python.exe")
$BackendPort = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { '8000' }
$FrontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { '3000' }

Set-Location $Root

if (-not (Test-Path $VenvPython)) {
    Write-Host '[Helios] venv not found. Running bootstrap first...'
    & (Join-Path $PSScriptRoot 'bootstrap.ps1')
}

if ((Test-Path (Join-Path $Root 'frontend\package.json')) -and -not (Test-Path (Join-Path $Root 'frontend\node_modules'))) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw '[Helios] npm is required to run the frontend.'
    }
    Write-Host '[Helios] Installing missing frontend dependencies...'
    Push-Location (Join-Path $Root 'frontend')
    npm install
    Pop-Location
}

$prepareAssets = if ($env:HELIOS_PREPARE_ASSETS) { $env:HELIOS_PREPARE_ASSETS } else { '1' }
if ($prepareAssets -eq '1') {
    Write-Host '[Helios] Preparing datasets, models, RAG assets, and GNN checkpoint...'
    & $VenvPython (Join-Path $Root 'scripts\prepare_startup_assets.py')
} else {
    Write-Host "[Helios] Skipping asset preparation (HELIOS_PREPARE_ASSETS=$prepareAssets)"
}

Write-Host "[Helios] Starting API on http://localhost:$BackendPort"
$backendProc = Start-Process -FilePath $VenvPython -ArgumentList '-m','uvicorn','helios.ui.api_server_v2:app','--host','0.0.0.0','--port',$BackendPort,'--reload' -PassThru

try {
    Write-Host "[Helios] Starting frontend on http://localhost:$FrontendPort"
    Push-Location (Join-Path $Root 'frontend')
    npm run dev -- --host 0.0.0.0 --port $FrontendPort
    Pop-Location
}
finally {
    if ($backendProc -and -not $backendProc.HasExited) {
        Write-Host '[Helios] Stopping API process...'
        Stop-Process -Id $backendProc.Id -Force
    }
}
