import locale
import os
import platform
import re
import shutil
import subprocess
import sys
import socket
import sysconfig
from importlib import metadata
from typing import List
from pathlib import Path
from typing import Optional

from mikazuki.log import log

python_bin = sys.executable

DEFAULT_ONNXRUNTIME_VERSION = "1.24.1"
LEGACY_GLIBC_ONNXRUNTIME_VERSION = "1.16.3"


def base_dir_path():
    return Path(__file__).parents[1].absolute()


def find_windows_git():
    possible_paths = ["git\\bin\\git.exe", "git\\cmd\\git.exe", "Git\\mingw64\\libexec\\git-core\\git.exe", "C:\\Program Files\\Git\\cmd\\git.exe"]
    for path in possible_paths:
        if os.path.exists(path):
            return path


def prepare_git():
    if shutil.which("git"):
        return True

    log.info("Finding git...")

    if sys.platform == "win32":
        git_path = find_windows_git()

        if git_path is not None:
            log.info(f"Git not found, but found git in {git_path}, add it to PATH")
            os.environ["PATH"] += os.pathsep + os.path.dirname(git_path)
            return True
        else:
            return False
    else:
        log.error("git not found, please install git first")
        return False


def prepare_submodules():
    frontend_path = base_dir_path() / "frontend" / "dist"
    tag_editor_path = base_dir_path() / "mikazuki" / "dataset-tag-editor" / "scripts"
    anima_trainer_path = base_dir_path() / "sd-scripts" / "anima_train_network.py"

    required_paths = [frontend_path, tag_editor_path, anima_trainer_path]
    if not all(os.path.exists(path) for path in required_paths):
        log.info("submodule not found, try clone...")
        log.info("checking git installation...")
        if not prepare_git():
            log.error("git not found, please install git first")
            sys.exit(1)
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=False)

    if not os.path.exists(anima_trainer_path):
        log.warning(
            "Anima trainer submodule is unavailable. Anima training will not work until "
            "`git submodule update --init --recursive` succeeds."
        )


def git_tag(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "describe", "--tags"], stderr=subprocess.DEVNULL).strip().decode("utf-8")
    except Exception:
        return "<none>"


def check_dirs(dirs: List):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def run(command,
        desc: Optional[str] = None,
        errdesc: Optional[str] = None,
        custom_env: Optional[list] = None,
        live: Optional[bool] = True,
        shell: Optional[bool] = None):

    if shell is None:
        shell = False if sys.platform == "win32" else True

    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=shell, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=shell, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def _installed_version(package_name: str) -> Optional[str]:
    candidates = (
        package_name,
        package_name.lower(),
        package_name.replace('_', '-'),
    )
    for candidate in dict.fromkeys(candidates):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def is_installed(package, friendly: str = None):
    #
    # This function was adapted from code written by vladmandic: https://github.com/vladmandic/automatic/commits/master
    #

    # Remove brackets and their contents from the line using regular expressions
    # e.g., diffusers[torch]==0.10.2 becomes diffusers==0.10.2
    package = re.sub(r'\[.*?\]', '', package)

    try:
        if friendly:
            pkgs = friendly.split()
        else:
            pkgs = [
                p
                for p in package.split()
                if not p.startswith('-') and not p.startswith('=')
            ]
            pkgs = [
                p.split('/')[-1] for p in pkgs
            ]   # get only package name if installing from URL

        for pkg in pkgs:
            if '>=' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('>=')]
            elif '==' in pkg:
                pkg_name, pkg_version = [x.strip() for x in pkg.split('==')]
            else:
                pkg_name, pkg_version = pkg.strip(), None

            version = _installed_version(pkg_name)

            if version is not None:
                if pkg_version is not None:
                    if '>=' in pkg:
                        ok = version >= pkg_version
                    else:
                        ok = version == pkg_version

                    if not ok:
                        log.info(f'Package wrong version: {pkg_name} {version} required {pkg_version}')
                        return False
            else:
                log.warning(f'Package version not found: {pkg_name}')
                return False

        return True
    except ModuleNotFoundError:
        log.warning(f'Package not installed: {pkgs}')
        return False


def validate_requirements(requirements_file: str):
    with open(requirements_file, 'r', encoding='utf8') as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() != ''
            and not line.startswith("#")
            and not (line.startswith("-") and not line.startswith("--index-url "))
            and line is not None
            and "# skip_verify" not in line
        ]

        index_url = ""
        for line in lines:
            if line.startswith("--index-url "):
                index_url = line.replace("--index-url ", "")
                continue

            if not is_installed(line):
                if index_url != "":
                    run_pip(f"install {line} --index-url {index_url}", line, live=True)
                else:
                    run_pip(f"install {line}", line, live=True)


def setup_windows_bitsandbytes():
    if sys.platform != "win32":
        return

    bnb_package = "bitsandbytes==0.46.0"
    bnb_path = os.path.join(sysconfig.get_paths()["purelib"], "bitsandbytes")

    installed_bnb = is_installed("bitsandbytes")  # don't check version here
    bnb_cuda_setup = (
        os.path.isdir(bnb_path)
        and any(re.findall(r"libbitsandbytes_cuda.+?\.dll", f) for f in os.listdir(bnb_path))
    )

    if not installed_bnb or not bnb_cuda_setup:
        log.error("detected wrong install of bitsandbytes, reinstall it")
        run_pip("uninstall bitsandbytes -y", "bitsandbytes", live=True)
        run_pip(f"install {bnb_package}", bnb_package, live=True)


def _torch_cuda_major() -> Optional[int]:
    try:
        import torch
        cuda_version = getattr(torch.version, "cuda", None)
    except Exception:
        return None

    if not cuda_version:
        return None

    try:
        return int(str(cuda_version).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _onnxruntime_target(
        onnx_version: Optional[str] = None,
        index_url: Optional[str] = None
):
    explicit_package = os.environ.get("ONNXRUNTIME_PACKAGE")
    if explicit_package and explicit_package not in {"onnxruntime", "onnxruntime-gpu"}:
        raise ValueError("ONNXRUNTIME_PACKAGE must be 'onnxruntime' or 'onnxruntime-gpu'")

    explicit_version = os.environ.get("ONNXRUNTIME_VERSION")
    target_version = explicit_version or onnx_version or DEFAULT_ONNXRUNTIME_VERSION
    target_index = os.environ.get("ONNXRUNTIME_INDEX_URL", index_url)

    if sys.platform == "linux":
        libc_name, libc_version = platform.libc_ver()
        if libc_name == "glibc" and libc_version and libc_version <= "2.27" and not explicit_version:
            target_version = LEGACY_GLIBC_ONNXRUNTIME_VERSION

    if explicit_package:
        return explicit_package, target_version, target_index

    cuda_major = _torch_cuda_major()
    if cuda_major is not None and cuda_major >= 12:
        return "onnxruntime-gpu", target_version, target_index

    if cuda_major is not None and cuda_major < 12:
        log.warning(
            "The default DTS ONNX Runtime GPU build targets CUDA 12.x; "
            "using CPU ONNX Runtime for this legacy CUDA environment. "
            "Set ONNXRUNTIME_PACKAGE/ONNXRUNTIME_VERSION/ONNXRUNTIME_INDEX_URL "
            "to override this expert fallback."
        )
    else:
        log.info("CUDA-enabled PyTorch not detected for ONNX Runtime; using the CPU package")

    return "onnxruntime", target_version, target_index


def setup_onnxruntime(
        onnx_version: Optional[str] = None,
        index_url: Optional[str] = None
):
    target_package, target_version, target_index = _onnxruntime_target(onnx_version, index_url)
    cpu_version = _installed_version("onnxruntime")
    gpu_version = _installed_version("onnxruntime-gpu")

    expected_version = cpu_version if target_package == "onnxruntime" else gpu_version
    other_version = gpu_version if target_package == "onnxruntime" else cpu_version

    if expected_version == target_version and other_version is None:
        return

    if cpu_version is not None or gpu_version is not None:
        log.info(
            "repairing ONNX Runtime installation: "
            f"cpu={cpu_version or 'none'}, gpu={gpu_version or 'none'}, "
            f"target={target_package}=={target_version}"
        )
        # The CPU and GPU distributions expose the same `onnxruntime` namespace.
        # Remove both before reinstalling one target package so shared files are
        # never left in a mixed/partially-overwritten state.
        run_pip("uninstall -y onnxruntime onnxruntime-gpu", "old ONNX Runtime packages", live=True)

    log.info(f"installing {target_package}=={target_version}")
    pip_install(target_package, target_version, index_url=target_index, live=True)


def run_pip(command, desc=None, live=False):
    return run(f'"{python_bin}" -m pip {command}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def pip_install(package: str, version: Optional[str] = None, index_url: Optional[str] = None, live: bool = True):
    """
    Install a package using pip.
    :param package: The name of the package to install.
    :param version: The version of the package to install (optional).
    :param index_url: The index URL to use for installing the package (optional).
    """
    if version:
        package = f"{package}=={version}"

    command = f"install {package}"

    if index_url:
        command = f"{command} -i {index_url}"

    run_pip(command, desc=f"Installing {package}", live=live)


def check_run(file: str) -> bool:
    result = subprocess.run([python_bin, file], capture_output=True, shell=False)
    log.info(result.stdout.decode("utf-8").strip())
    return result.returncode == 0


def network_gfw_test(timeout=3):
    try:
        import requests
        # requests will auto detect system proxies
        response = requests.get("https://www.google.com", timeout=timeout)
        if response.status_code == 200:
            log.info("Network test passed")
            return True
        else:
            log.error(f"Network test failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        log.error(f"Network test failed: {e}")
        return False


def prepare_environment(disable_auto_mirror: bool = True, prepare_onnxruntime: bool = True):
    if sys.platform == "win32":
        # disable triton on windows
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    if not disable_auto_mirror and not network_gfw_test():
        log.info("use pip & huggingface mirrors")
        os.environ.setdefault("PIP_FIND_LINKS", "https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html")
        os.environ.setdefault("PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if not os.environ.get("PATH"):
        os.environ["PATH"] = os.path.dirname(sys.executable)

    prepare_submodules()

    check_dirs(["config/autosave", "logs"])

    validate_requirements("requirements.txt")
    setup_windows_bitsandbytes()

    if prepare_onnxruntime:
        setup_onnxruntime()


def catch_exception(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            log.error(f"An error occurred: {e}")
    return wrapper


def check_port_avaliable(port: int):
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except Exception:
        return False


def find_avaliable_ports(port_init: int, port_range: int):
    server_ports = range(port_init, port_range)

    for p in server_ports:
        if check_port_avaliable(p):
            return p

    log.error(f"error finding avaliable ports in range: {port_init} -> {port_range}")
    return None
