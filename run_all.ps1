$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python .\run_all.py --profile full --resume --verify
exit $LASTEXITCODE
