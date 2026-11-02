"""cuda_attention_backend.py - use our CUDA attention kernels inside the GPT.

The kernels live in ../triton_attention/cuda_ext/ and were copied VERBATIM from
cuda_attention/benchmark.cu. We build them here as a PyTorch extension and expose a
single drop-in function, `cuda_causal_attention`, that the GPT's attention head can
call instead of the usual PyTorch `softmax(QK^T/sqrt(d)) @ V`.

Important properties of these kernels (they decide how we integrate):
  * FP32 only
  * built-in CAUSAL mask (token i attends to tokens 0..i)
  * scaling is 1/sqrt(head_dim) -- the standard transformer scale
  * FORWARD-ONLY: there is no backward kernel, so this backend is used for
    inference / generation, not for training. Training uses the PyTorch backend.

The C++ entry point (triton_attention/cuda_ext/attention_ext.cpp) is:
    attention(q, k, v, scores, out, optimized)
where q, k, v are 2-D contiguous FP32 CUDA tensors [T, head_dim], `scores` is a
preallocated [T, T] buffer, `out` is [T, head_dim], and `optimized` chooses the
shared-memory tiled Q@K^T kernel (True) or the naive one (False). We therefore loop
over (batch, head), calling the kernel once per [T, head_dim] head.
"""

import functools
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_EXT_DIR = os.path.join(_REPO, "triton_attention", "cuda_ext")
_EXT_SOURCES = [
    os.path.join(_EXT_DIR, "attention_ext.cpp"),
    os.path.join(_EXT_DIR, "attention_kernels.cu"),
]


def _nvcc_major():
    """Major CUDA version of nvcc on PATH (e.g. 13), or None if nvcc is absent."""
    import re
    import shutil
    import subprocess
    exe = shutil.which("nvcc")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=15).stdout
        m = re.search(r"release (\d+)\.", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    """Compile (once) and return the CUDA attention extension module.

    Raises a RuntimeError with a readable hint if something is missing. The first
    call is slow (nvcc + cl compile); afterwards it is cached by torch.
    """
    for src in _EXT_SOURCES:
        if not os.path.exists(src):
            raise FileNotFoundError(f"CUDA kernel source not found: {src}")

    # ---- Pre-flight checks (clearer than torch's own build errors) ----------
    try:
        import setuptools  # noqa: F401  (torch.utils.cpp_extension needs it)
    except ImportError as exc:
        raise RuntimeError(
            "'setuptools' is required to build the CUDA extension but is not installed "
            "in this Python environment.\n  Fix: pip install setuptools"
        ) from exc

    torch_cuda = torch.version.cuda
    if torch_cuda is None:
        raise RuntimeError(
            "This PyTorch build has no CUDA support (CPU-only wheel). Install a CUDA build "
            "of torch — see triton_attention/SETUP.md."
        )
    torch_major = int(torch_cuda.split(".")[0])
    nvcc_major = _nvcc_major()
    if nvcc_major is not None and nvcc_major != torch_major:
        raise RuntimeError(
            f"CUDA major-version mismatch: PyTorch {torch.__version__} is built for CUDA "
            f"{torch_cuda} (major {torch_major}), but nvcc is CUDA {nvcc_major}.x. "
            "torch.utils.cpp_extension requires the SAME major version to build a CUDA "
            "extension.\n"
            f"  Fix: run in the .venv-triton environment (torch built for cu{nvcc_major}0), "
            "which matches nvcc — see triton_attention/SETUP.md. The current environment's "
            "torch cannot build against this nvcc."
        )

    # ---- Build ---------------------------------------------------------------
    try:
        from torch.utils.cpp_extension import load
        return load(
            name="attention_cuda_ext",
            sources=_EXT_SOURCES,
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            extra_cuda_cflags=["-O3"],   # no --use_fast_math: keep expf accurate
            verbose=verbose,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build the CUDA attention extension.\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "On Windows this usually means the build toolchain is not on PATH.\n"
            "Fix: run from the 'x64 Native Tools Command Prompt for VS 2022' so cl.exe is\n"
            "found, with nvcc (CUDA toolkit) on PATH, inside the .venv-triton env whose\n"
            "torch CUDA major version matches nvcc (see triton_attention/SETUP.md)."
        ) from exc


def is_available() -> tuple[bool, str]:
    """(True, '') if the extension builds and a CUDA GPU is present, else (False, reason)."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False (CPU-only PyTorch build or no GPU)."
    try:
        load_extension()
        return True, ""
    except Exception as exc:
        return False, str(exc)


@torch.no_grad()
def cuda_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          optimized: bool = True) -> torch.Tensor:
    """Causal self-attention computed by our CUDA kernels.

    q, k, v : (B, n_head, T, head_dim)  or  (B, T, head_dim)  -- FP32 CUDA tensors.
    Returns a tensor of the same shape as the input.

    We loop over (batch, head) because each kernel invocation handles one
    [T, head_dim] head. This is for correctness / integration, not peak speed;
    the headline speed numbers come from the dedicated benchmarks.
    """
    ext = load_extension()

    added_head_dim = (q.dim() == 3)
    if added_head_dim:                       # (B, T, hd) -> (B, 1, T, hd)
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)

    if q.dim() != 4:
        raise ValueError(f"expected q of shape (B, nh, T, hd) or (B, T, hd), got {tuple(q.shape)}")

    B, nh, T, hd = q.shape
    q = q.to(dtype=torch.float32).contiguous()
    k = k.to(dtype=torch.float32).contiguous()
    v = v.to(dtype=torch.float32).contiguous()

    out = torch.empty_like(q)
    scores = torch.empty(T, T, device=q.device, dtype=torch.float32)   # reused buffer
    obuf = torch.empty(T, hd, device=q.device, dtype=torch.float32)    # reused buffer

    for b in range(B):
        for h in range(nh):
            ext.attention(
                q[b, h].contiguous(), k[b, h].contiguous(), v[b, h].contiguous(),
                scores, obuf, optimized,
            )
            out[b, h] = obuf

    return out.squeeze(1) if added_head_dim else out
