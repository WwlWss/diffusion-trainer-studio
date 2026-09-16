$repoRoot = $PSScriptRoot
if (-not $Env:HF_HUB_CACHE) {
    $Env:HF_HUB_CACHE = Join-Path $repoRoot "huggingface\hub"
}

if (!(Test-Path -Path "venv")) {
    Write-Output "Creating venv for Python..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "Installing deps..."

python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -U -I --no-deps xformers==0.0.30 --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade -r requirements.txt

Write-Output "Checking dependency consistency..."
python -m pip check
if (-not $?) {
    Write-Error "Dependency consistency check failed. Please review the pip output above."
    exit 1
}

Write-Output "Install completed"
Read-Host | Out-Null
