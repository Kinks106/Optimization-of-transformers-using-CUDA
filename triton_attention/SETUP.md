# Triton setup (Windows 11, RTX 4060 / sm_89, CUDA 13.3, MSVC 2022)

Recommended path: **native Windows with the community `triton-windows` package.**

Why native Windows instead of WSL2:

- Our CUDA kernels are already built with MSVC + CUDA 13.3 on Windows. In Step 4 they are
  compiled as a PyTorch extension and timed in the *same Python process* as the Triton
  kernels, on the same driver. That only works if Triton and the extension share one OS.
- `triton-windows` wheels bundle what Triton needs to JIT-compile kernels (a minimal CUDA
  toolchain with `ptxas`, plus the TinyCC C compiler), so `triton.jit` works without extra setup.
- WSL2 would need a Linux distro, a second CUDA toolkit inside it, and gcc for the extension.
  It is the fallback if `triton-windows` fails (commands at the bottom).

Trade-off: `triton-windows` is a community fork (github.com/woct0rdho/triton-windows),
not an official OpenAI/Triton release. It tracks upstream closely and is widely used
(ComfyUI, SageAttention). We use it unchanged; the kernel code is identical to upstream Triton.

## Versions

| Package | Version | Why |
|---|---|---|
| Python | 3.12 or 3.13 (python.org, **not** Microsoft Store) | `triton-windows` needs >= 3.11 |
| torch | 2.14.1 + cu130 | CUDA 13.x build: Step 4's extension needs torch's CUDA **major** version (13) to match our nvcc 13.3 |
| triton-windows | 3.8.x | Triton 3.8 is the version torch 2.14 is built against |

torch 2.15.0 (cu132) and triton-windows 3.9.0 also exist, but both came out on 28 Oct 2026.
The 2.14.1 / 3.8 pair has been out for about a month, so we pin that.

Do **not** reuse the repo's `requirements.txt` environment for this. It installs torch from
the `cu121` index (CUDA 12.1), and `torch.utils.cpp_extension` will refuse to build with
nvcc 13.3 because the CUDA major versions differ (12 vs 13). Use a separate venv:

## Install (PowerShell, from the repo root)

```powershell
# 0. Prerequisites (once): up-to-date NVIDIA driver, and the VC++ runtime 14.42+
#    (triton-windows' libtriton.pyd is built with MSVC). Skip winget if already installed.
winget install Microsoft.VCRedist.2015+.x64

# 1. Separate virtual environment (calls python.exe directly, so no Activate.ps1 / execution-policy change is needed)
py -3.13 -m venv .venv-triton
.\.venv-triton\Scripts\python.exe -m pip install --upgrade pip

# 2. PyTorch with the CUDA 13.0 runtime
.\.venv-triton\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# 3. Triton for Windows (remove any stray 'triton' package first)
.\.venv-triton\Scripts\python.exe -m pip uninstall -y triton
.\.venv-triton\Scripts\python.exe -m pip install "triton-windows>=3.8,<3.9"

# 4. Extras: ninja is required by torch.utils.cpp_extension (Step 4); numpy silences torch warnings
.\.venv-triton\Scripts\python.exe -m pip install ninja numpy

# 5. Smoke test
.\.venv-triton\Scripts\python.exe triton_attention\00_vector_add.py
```

## Benchmark (Step 4: builds our CUDA kernels as a PyTorch extension)

Run from an **"x64 Native Tools Command Prompt for VS 2022"** (or after `vcvars64.bat`) so
`cl.exe` and its headers are found, with `CUDA_PATH` pointing at CUDA 13.3 so `nvcc` is found:

```bat
.venv-triton\Scripts\python.exe triton_attention\check_correctness.py
.venv-triton\Scripts\python.exe triton_attention\benchmark_triton.py
```

For stable numbers: plug in the charger, close other GPU programs, and use the Windows
"Best performance" power mode. Results are written to `triton_attention\results\`.

**No system CUDA toolkit?** (This is how the RTX 3050 dev laptop built it.) `nvcc` from pip also
works, as long as its CUDA major version (13) matches torch's:

```bat
.venv-triton\Scripts\python.exe -m pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-runtime==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*"
set CUDA_HOME=%CD%\.venv-triton\Lib\site-packages\nvidia\cu13
set PATH=%CUDA_HOME%\bin;%PATH%
```

Pin all five packages to 13.0: without the pins, pip installs `crt`/`nvvm` 13.4, which do not match `nvcc` 13.0.

## Troubleshooting (Windows)

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | A CPU wheel was installed. Reinstall with the `--index-url .../cu130` line. |
| `ImportError: DLL load failed while importing libtriton` | VC++ runtime missing/old. Install it (step 0). |
| `DLL load failed while importing libtriton: The filename or extension is too long` | The venv's full path is too long (Windows' 260-character limit; seen on the dev laptop). Put the venv on a short path, e.g. `py -3.13 -m venv C:\venvs\triton`, and use `C:\venvs\triton\Scripts\python.exe` in the commands above. |
| Extension build: `Cannot open include file: 'detail/class.h'` or `'math_functions.hpp'` although the file exists | Same 260-character limit: `cl.exe` and `nvcc` do not support long paths even when Windows long-path support is on. Use a venv whose path is short (the repo-local `.venv-triton` is fine unless the repo itself is deeply nested). |
| Extension build: `'vswhere.exe' is not recognized` | Harmless message from `vcvars64.bat`; ignore it. |
| `ImportError: DLL load failed while importing cuda_utils` | Stale cache. Delete `%USERPROFILE%\.triton\cache\` and retry. |
| Errors after changing Python/Triton/CUDA version | Same: delete `%USERPROFILE%\.triton\cache\`. |
| `ModuleNotFoundError: triton` while a file/folder named `triton` is nearby | Don't name scripts `triton.py` or folders `triton/`. |
| Crash with `Exception Code: 0x80000003` | Usually an earlier CUDA error. Rerun with `$env:CUDA_LAUNCH_BLOCKING=1` to see it. |

## Fallback: WSL2 + official Triton

Use this only if the Windows path fails. Note the error first, so it can go in the report.

```powershell
wsl --install -d Ubuntu-24.04
```

Inside Ubuntu (do **not** install a Linux NVIDIA driver; WSL uses the Windows driver):

```bash
sudo apt update && sudo apt install -y python3-venv build-essential
# CUDA toolkit for WSL (needed only for Step 4's extension)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install -y cuda-toolkit-13-0
python3 -m venv ~/triton-env && source ~/triton-env/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # pulls official triton 3.8 automatically
pip install ninja numpy
python triton_attention/00_vector_add.py
```
