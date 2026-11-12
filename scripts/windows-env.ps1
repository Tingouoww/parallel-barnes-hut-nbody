# Dot-source this file in PowerShell to use the portable tools installed for
# this checkout. It changes only the current shell's environment.
#   . .\scripts\windows-env.ps1
#   mingw32-make -j4
#   python tests\test_cli.py
param(
    [string]$ToolsRoot = (Join-Path $PSScriptRoot '..\..\.tools')
)
$compilerBin = Join-Path $ToolsRoot 'winlibs\mingw64\bin'
$pythonBin = Join-Path $ToolsRoot 'analysis-env\Scripts'
$gitBashBin = 'C:\Program Files\Git\usr\bin'
if (-not (Test-Path -LiteralPath (Join-Path $compilerBin 'g++.exe'))) {
    throw "Portable compiler not found under $ToolsRoot. Pass -ToolsRoot for your installation."
}
$env:Path = "$compilerBin;$pythonBin;$gitBashBin;" + $env:Path
$env:PYTHON = 'python'
