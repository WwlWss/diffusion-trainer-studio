$repoRoot = $PSScriptRoot
if (-not $Env:HF_HUB_CACHE) {
    $Env:HF_HUB_CACHE = Join-Path $repoRoot "huggingface\hub"
}
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

function InstallFail {
    param([string]$Message = "安装失败。")
    Write-Output $Message
    Read-Host | Out-Null
    Exit 1
}

function Assert-NativeSuccess {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) {
        InstallFail $Message
    }
}

function Get-PythonMinor {
    param([string]$PythonExe)
    return (& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
}

function Assert-CompatiblePython {
    param([string]$PythonExe)
    $version = Get-PythonMinor $PythonExe
    if ($LASTEXITCODE -ne 0 -or $version -notin @("3.10", "3.11", "3.12")) {
        InstallFail "不支持 Python $version。DTS v2.0.0 安装器支持 Python 3.10-3.12，其中 Python 3.11 为当前 CI 与发布测试版本。"
    }
    if ($version -ne "3.11") {
        Write-Warning "当前 Python $version 可由安装器使用，但 DTS v2.0.0 当前推荐并测试 Python 3.11。"
    }
}

if (Test-Path -Path "python\python.exe") {
    Write-Output "使用 python 文件夹内的 python..."
    Assert-CompatiblePython ".\python\python.exe"
    $py_path = (Get-Item "python").FullName
    $env:PATH = "$py_path;$env:PATH"
}
else {
    if (!(Test-Path -Path "venv")) {
        Write-Output "正在创建虚拟环境..."
        $py311 = Get-Command py -ErrorAction SilentlyContinue
        if ($py311) {
            py -3.11 -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                py -3.11 -m venv venv
                Assert-NativeSuccess "创建 Python 3.11 虚拟环境失败。"
            }
            else {
                Assert-CompatiblePython "python"
                python -m venv venv
                Assert-NativeSuccess "创建虚拟环境失败。"
            }
        }
        else {
            Assert-CompatiblePython "python"
            python -m venv venv
            Assert-NativeSuccess "创建虚拟环境失败。"
        }
    }

    Assert-CompatiblePython ".\venv\Scripts\python.exe"
    Write-Output "检测到虚拟环境，尝试激活..."
    .\venv\Scripts\activate
    if (-not $?) {
        InstallFail "激活虚拟环境失败。"
    }
}

Write-Output "安装程序所需依赖 (已进行国内加速，若在国外或无法使用加速源请换用 install.ps1 脚本)"
Write-Output "受限于国内加速镜像，torch 安装无法使用镜像源，安装较为缓慢。"
$install_torch = Read-Host "是否需要安装 Torch+xformers? [y/n] (默认为 y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq "") {
    python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    Assert-NativeSuccess "torch 安装失败，请删除 venv 文件夹后重新运行。"
    python -m pip install -U -I --no-deps xformers==0.0.30 --extra-index-url https://download.pytorch.org/whl/cu128
    Assert-NativeSuccess "xformers 安装失败。"
}

python -m pip install --upgrade -r requirements.txt
Assert-NativeSuccess "训练界面依赖安装失败。"

Write-Output "检查依赖一致性..."
python -m pip check
Assert-NativeSuccess "依赖一致性检查失败，请查看上方 pip 输出。"

Write-Output "安装完毕"
Read-Host | Out-Null
