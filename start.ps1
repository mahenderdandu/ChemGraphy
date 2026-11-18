$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Root 'scripts\start.ps1')
