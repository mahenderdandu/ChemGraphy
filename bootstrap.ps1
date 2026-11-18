$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Root 'scripts\bootstrap.ps1')
