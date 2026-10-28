"""
00_vector_add.py - Triton smoke test (Step 1).

1. Prints the environment (Python, PyTorch, CUDA, Triton, GPU).
2. Runs Triton's classic vector-add kernel and checks it against PyTorch.
3. Checks that tl.dot can run in full FP32 (input_precision="ieee", i.e. no TF32).
   The attention kernels in this folder depend on this, because our CUDA code is FP32.

Run from the repo root:
    python triton_attention/00_vector_add.py

Exit code 0 = all checks passed.
"""

import os
import platform
import sys

import torch
import triton
import triton.language as tl

# TRITON_INTERPRET=1 runs kernels on the CPU (debug mode, no GPU needed).
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"


# ----------------------------------------------------------------------------
# Kernel 1: vector add (Triton tutorial 01)
# ----------------------------------------------------------------------------
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # One "program" (like one CUDA block) handles BLOCK_SIZE elements at once.
    # There is no threadIdx: we operate on a whole block of offsets as a vector.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # guard the last, partial block
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


# ----------------------------------------------------------------------------
# Kernel 2: one small tl.dot, to check FP32 ("ieee") vs TF32 precision
# ----------------------------------------------------------------------------
@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               PRECISION: tl.constexpr):
    # A single program computes the whole (M x N) = (M x K) @ (K x N) product.
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b, input_precision=PRECISION)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


def triton_dot(a: torch.Tensor, b: torch.Tensor, precision: str) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    dot_kernel[(1,)](a, b, c, M=M, N=N, K=K, PRECISION=precision)
    return c


# ----------------------------------------------------------------------------
def print_environment() -> bool:
    print("=" * 60)
    print(" TRITON SMOKE TEST")
    print("=" * 60)
    print(f"OS            : {platform.platform()}")
    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__} (built for CUDA {torch.version.cuda})")
    print(f"Triton        : {triton.__version__}")
    if INTERPRET:
        print("Mode          : TRITON_INTERPRET=1 (CPU interpreter, no GPU)")
        return True
    if not torch.cuda.is_available():
        print("\nFAIL: torch.cuda.is_available() is False.")
        print("      This PyTorch is a CPU-only build. Reinstall with:")
        print("      pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130")
        return False
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU           : {torch.cuda.get_device_name(0)} (sm_{major}{minor})")
    return True


def check_vector_add() -> bool:
    print("\n[1] Vector add")
    torch.manual_seed(0)
    ok = True
    # Sizes include ones that are not a multiple of BLOCK_SIZE, to test the mask.
    for n in (1, 1000, 98_432, 1 << 20):
        x = torch.rand(n, device=DEVICE)
        y = torch.rand(n, device=DEVICE)
        out = triton_add(x, y)
        same = torch.equal(out, x + y)
        ok &= same
        print(f"    n = {n:>9,}   matches torch: {same}")

    if not INTERPRET:
        n = 1 << 24  # 16M floats = 64 MB per tensor
        x = torch.rand(n, device=DEVICE)
        y = torch.rand(n, device=DEVICE)
        ms_triton = triton.testing.do_bench(lambda: triton_add(x, y))
        ms_torch = triton.testing.do_bench(lambda: x + y)
        gbps = lambda ms: 3 * n * 4 / (ms * 1e-3) / 1e9  # read x, read y, write out
        print(f"    n = 16,777,216 timing (do_bench, median):")
        print(f"        Triton : {ms_triton:.4f} ms  ({gbps(ms_triton):.1f} GB/s)")
        print(f"        PyTorch: {ms_torch:.4f} ms  ({gbps(ms_torch):.1f} GB/s)")
    return ok


def check_fp32_dot() -> bool:
    print("\n[2] tl.dot precision (64x64 @ 64x64, FP32 inputs)")
    torch.manual_seed(0)
    a = torch.rand(64, 64, device=DEVICE) * 2 - 1  # same range as our CUDA inputs: [-1, 1]
    b = torch.rand(64, 64, device=DEVICE) * 2 - 1
    ref = (a.double().cpu() @ b.double().cpu())   # float64 reference

    err_ieee = (triton_dot(a, b, "ieee").double().cpu() - ref).abs().max().item()
    err_tf32 = (triton_dot(a, b, "tf32").double().cpu() - ref).abs().max().item()
    print(f"    max |error| with input_precision='ieee' (full FP32): {err_ieee:.3e}")
    print(f"    max |error| with input_precision='tf32'            : {err_tf32:.3e}")

    ok = err_ieee < 1e-4
    if INTERPRET:
        print("    (interpreter mode computes both in FP32, so they match; run on the GPU to see TF32)")
    elif err_tf32 <= err_ieee:
        print("    note: TF32 was not less accurate than IEEE here; unexpected on sm_80+")
    print(f"    full-FP32 dot accurate: {ok}")
    return ok


def main() -> int:
    if not print_environment():
        return 1
    ok_add = check_vector_add()
    ok_dot = check_fp32_dot()
    print("\n" + "=" * 60)
    if ok_add and ok_dot:
        print(" PASS: Triton compiles and runs kernels, and FP32 tl.dot works.")
        print("=" * 60)
        return 0
    print(" FAIL: see the checks above.")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
