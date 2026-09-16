$repoRoot = $PSScriptRoot
if (-not $Env:HF_HUB_CACHE) {
    $Env:HF_HUB_CACHE = Join-Path $repoRoot "huggingface\hub"
}
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

function InstallFail {
    Write-Output "安装失败。"
    Read-Host | Out-Null
    Exit 1
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

if (Test-Path -Path "python\python.exe") {
    Write-Output "使用 python 文件夹内的 python..."
    $py_path = (Get-Item "python").FullName
    $env:PATH = "$py_path;$env:PATH"
}
else {
    if (!(Test-Path -Path "venv")) {
        Write-Output "正在创建虚拟环境..."
        python -m venv venv
        Check "创建虚拟环境失败。DTS v2.0.0 当前推荐并测试 Python 3.11 64 位，请确认 Python 已安装并加入 PATH。"
    }

    Write-Output "检测到虚拟环境，尝试激活..."
    .\venv\Scripts\activate
    Check "激活虚拟环境失败。"
}

Write-Output "安装程序所需依赖 (已进行国内加速，若在国外或无法使用加速源请换用 install.ps1 脚本)"
Write-Output "受限于国内加速镜像，torch 安装无法使用镜像源，安装较为缓慢。"
$install_torch = Read-Host "是否需要安装 Torch+xformers? [y/n] (默认为 y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq "") {
    python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    Check "torch 安装失败，请删除 venv 文件夹后重新运行。"
    python -m pip install -U -I --no-deps xformers==0.0.30 --extra-index-url https://download.pytorch.org/whl/cu128
    Check "xformers 安装失败。"
}

python -m pip install --upgrade -r requirements.txt
Check "训练界面依赖安装失败。"

Write-Output "检查依赖一致性..."
python -m pip check
Check "依赖一致性检查失败，请查看上方 pip 输出。"

Write-Output "安装完毕"
Read-Host | Out-Null
