$repoRoot = $PSScriptRoot
if (-not $Env:HF_HUB_CACHE) {
    $Env:HF_HUB_CACHE = Join-Path $repoRoot "huggingface\hub"
}

function Get-PythonMinor {
    param([string]$PythonExe)
    return (& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
}

function Assert-CompatiblePython {
    param([string]$PythonExe)
    $version = Get-PythonMinor $PythonExe
    if ($LASTEXITCODE -ne 0 -or $version -notin @("3.10", "3.11", "3.12")) {
        Write-Error "Unsupported Python '$version'. DTS v2.0.0 supports Python 3.10-3.12; Python 3.11 is the tested/recommended version."
        exit 1
    }
    if ($version -ne "3.11") {
        Write-Warning "Python $version is supported by the installer, but DTS v2.0.0 CI and release testing use Python 3.11."
    }
}

if (!(Test-Path -Path "venv")) {
    Write-Output "Creating venv for Python..."
    $py311 = Get-Command py -ErrorAction SilentlyContinue
    if ($py311) {
        py -3.11 -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) {
            py -3.11 -m venv venv
        }
        else {
            Assert-CompatiblePython "python"
            python -m venv venv
        }
    }
    else {
        Assert-CompatiblePython "python"
        python -m venv venv
    }
    if (-not $?) {
        Write-Error "Failed to create the virtual environment."
        exit 1
    }
}

Assert-CompatiblePython ".\venv\Scripts\python.exe"
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
