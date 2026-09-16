#!/usr/bin/bash
set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
create_venv=true

while [ -n "$1" ]; do
    case "$1" in
        --disable-venv)
            create_venv=false
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if $create_venv; then
    echo "Creating python venv..."
    python3 -m venv venv
    source "$script_dir/venv/bin/activate"
    echo "active venv"
fi

echo "Installing torch & xformers..."

cuda_version=$(nvidia-smi | grep -oiP 'CUDA Version: \K[\d\.]+' || true)

if [ -z "$cuda_version" ]; then
    cuda_version=$(nvcc --version 2>/dev/null | grep -oiP 'release \K[\d\.]+' || true)
fi

if [ -z "$cuda_version" ]; then
    echo "Unable to detect CUDA. DTS v2.0.0 requires a supported NVIDIA CUDA environment."
    exit 1
fi

cuda_major_version=$(echo "$cuda_version" | awk -F'.' '{print $1}')
cuda_minor_version=$(echo "$cuda_version" | awk -F'.' '{print $2}')

echo "CUDA Version: $cuda_version"

if (( cuda_major_version >= 12 )); then
    echo "install torch 2.7.0+cu128"
    python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    python -m pip install --no-deps xformers==0.0.30 --extra-index-url https://download.pytorch.org/whl/cu128
elif (( cuda_major_version == 11 && cuda_minor_version >= 8 )); then
    echo "install torch 2.4.0+cu118"
    python -m pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    python -m pip install --no-deps xformers==0.0.27.post2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
else
    echo "Unsupported CUDA version: $cuda_version"
    echo "DTS v2.0.0 supports CUDA 11.8 or CUDA 12+ through this installer."
    echo "The old CUDA 11.2/11.6 + PyTorch 1.12 branches are not compatible with the current DTS dependency stack."
    exit 1
fi

echo "Installing deps..."

cd "$script_dir" || exit 1
python -m pip install --upgrade -r requirements.txt

echo "Checking dependency consistency..."
python -m pip check

echo "Install completed"
