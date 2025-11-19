# --- run_setup.ps1 ---
# Purpose: Set up environment and run the BSML backtest runner
# Author: Vincenzo (P3 – Infrastructure)

# Step 1 – Compute absolute path for src
$srcPath = (Resolve-Path .\src).Path
$env:PYTHONPATH = $srcPath
Write-Host "`nPYTHONPATH set to:" -ForegroundColor Cyan
Write-Host $srcPath -ForegroundColor Yellow

# Step 2 – Quick import check
try {
    python -c "import importlib; m=importlib.import_module('bsml.policies.baseline'); print('✓ Imported:', m.__name__)"
}
catch {
    Write-Host "⚠️  Could not import baseline module. Check that src\bsml\policies\baseline.py exists." -ForegroundColor Red
    exit
}

# Step 3 – Run the runner
Write-Host "`nRunning backtest..." -ForegroundColor Cyan
python -m bsml.core.runner
Write-Host "`nFinished at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
