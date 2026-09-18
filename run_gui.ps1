$repoRoot = $PSScriptRoot

# Keep large Hub downloads inside the DTS checkout without relocating the user's
# Hugging Face authentication token. This lets a normal `hf auth login` remain
# valid across fresh DTS clones while preserving the existing local model cache.
if (-not $Env:HF_HUB_CACHE) {
    $Env:HF_HUB_CACHE = Join-Path $repoRoot "huggingface\hub"
}

# Preserve authentication created by older DTS/SD-Trainer versions that stored
# the token under the repository-local HF_HOME.
$legacyToken = Join-Path $repoRoot "huggingface\token"
if (-not $Env:HF_TOKEN_PATH -and (Test-Path $legacyToken)) {
    $Env:HF_TOKEN_PATH = $legacyToken
}

$Env:PYTHONUTF8 = "1"

if (Test-Path -Path "venv\Scripts\activate") {
    Write-Host -ForegroundColor green "Activating virtual environment..."
    .\venv\Scripts\activate
}
elseif (Test-Path -Path "python\python.exe") {
    Write-Host -ForegroundColor green "Using python from python folder..."
    $py_path = (Get-Item "python").FullName
    $env:PATH = "$py_path;$env:PATH"
}
else {
    Write-Host -ForegroundColor Blue "No virtual environment found, using system python..."
}

python gui.py
