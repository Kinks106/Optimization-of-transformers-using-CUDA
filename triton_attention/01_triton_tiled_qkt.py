"""
01_triton_tiled_qkt.py - Triton version of our optimized CUDA attention.

Same three-stage algorithm as cuda_attention/benchmark.cu (optimized path):

    Stage 1  scores = Q @ K^T / sqrt(d) + causal mask   (16x16 tiles, loop over d in 16-wide chunks)
    Stage 2  P = row-wise softmax(scores)                (scores live in global memory, N x N)
    Stage 3  O = P @ V                                   (only columns j <= row are read)

What is kept the same as the CUDA code, so the comparison is algorithm-for-algorithm:
  * FP32 inputs, FP32 math. tl.dot uses input_precision="ieee" (no TF32).
  * Row-major [N, d] single-head layout, full N x N scores matrix in global memory.
  * 16x16 output tiles for Q @ K^T, d walked in 16-wide chunks (same as TILE = 16).
  * All score tiles are computed, then masked (the CUDA kernel does not skip masked tiles).
  * Scale applied as a division by sqrt(d); masked entries set to -inf.
  * P @ V stops at the causal boundary (CUDA: `for j <= row`; here: up to the end of the row tile).

Known differences (by design of each language, mentioned in the report):
  * CUDA softmax runs one thread per row, serially. A Triton program works on whole
    blocks, so one program per row processes the row as a vector. Same 3-pass math
    (max, exp-sum, normalize), but parallel within the row.
  * In CUDA each thread computes one output element. In Triton each program computes a
    whole 16x16 tile with tl.dot; the compiler decides the thread mapping.
  * tl.exp uses the fast hardware exp2 path; CUDA's expf (no --use_fast_math) is
    slightly more accurate. The difference is around 1e-7 relative.
"""

import math

import torch
import triton
import triton.language as tl

TILE = 16  # same as `const int TILE = 16;` in the CUDA code


# ----------------------------------------------------------------------------
# Stage 1: tiled Q @ K^T, scale, causal mask
# ----------------------------------------------------------------------------
@triton.jit
def qkt_scores_kernel(
    q_ptr, k_ptr, s_ptr,
    N, D, sqrt_d,
    stride_q, stride_k, stride_s,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    # program_id(0) ~ blockIdx.x (column tile), program_id(1) ~ blockIdx.y (row tile)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):  # like `for tileStart ... tileStart += TILE`
        ks = k0 + tl.arange(0, BLOCK_K)
        # Q tile [BLOCK_M, BLOCK_K]
        q = tl.load(q_ptr + rows[:, None] * stride_q + ks[None, :],
                    mask=(rows[:, None] < N) & (ks[None, :] < D), other=0.0)
        # K^T tile [BLOCK_K, BLOCK_N]: K is read "transposed" by the index math, as in CUDA
        kt = tl.load(k_ptr + cols[None, :] * stride_k + ks[:, None],
                     mask=(cols[None, :] < N) & (ks[:, None] < D), other=0.0)
        acc = tl.dot(q, kt, acc, input_precision=PRECISION)

    s = acc / sqrt_d
    s = tl.where(cols[None, :] > rows[:, None], float("-inf"), s)  # causal mask
    tl.store(s_ptr + rows[:, None] * stride_s + cols[None, :], s,
             mask=(rows[:, None] < N) & (cols[None, :] < N))


# ----------------------------------------------------------------------------
# Stage 2: row-wise softmax (one program per row)
# ----------------------------------------------------------------------------
@triton.jit
def softmax_rows_kernel(s_ptr, N, stride_s, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(s_ptr + row * stride_s + cols, mask=cols < N, other=float("-inf"))
    x_max = tl.max(x, axis=0)          # pass 1: max
    e = tl.exp(x - x_max)              # pass 2: exp (masked -inf -> 0) ...
    p = e / tl.sum(e, axis=0)          # ... sum, pass 3: normalize
    tl.store(s_ptr + row * stride_s + cols, p, mask=cols < N)


# ----------------------------------------------------------------------------
# Stage 3: O = P @ V, tiled 16x16 over (rows, d), stopping at the causal boundary
# ----------------------------------------------------------------------------
@triton.jit
def pv_kernel(
    p_ptr, v_ptr, o_ptr,
    N, D,
    stride_p, stride_v, stride_o,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    PRECISION: tl.constexpr,
):
    # program_id(0) ~ blockIdx.x (d tile), program_id(1) ~ blockIdx.y (row tile)
    pid_m = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ds = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N)  # columns beyond this tile's last row are all 0
    for n0 in range(0, hi, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        p = tl.load(p_ptr + rows[:, None] * stride_p + cols[None, :],
                    mask=(rows[:, None] < N) & (cols[None, :] < N), other=0.0)
        v = tl.load(v_ptr + cols[:, None] * stride_v + ds[None, :],
                    mask=(cols[:, None] < N) & (ds[None, :] < D), other=0.0)
        acc = tl.dot(p, v, acc, input_precision=PRECISION)

    tl.store(o_ptr + rows[:, None] * stride_o + ds[None, :], acc,
             mask=(rows[:, None] < N) & (ds[None, :] < D))


# ----------------------------------------------------------------------------
# Python wrappers (host code)
# ----------------------------------------------------------------------------
def _check(x: torch.Tensor) -> None:
    assert x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous(), \
        "expected a contiguous float32 CUDA tensor of shape [N, d]"


def qkt_scores(q: torch.Tensor, k: torch.Tensor, scores: torch.Tensor,
               precision: str = "ieee") -> torch.Tensor:
    N, D = q.shape
    grid = (triton.cdiv(N, TILE), triton.cdiv(N, TILE))
    qkt_scores_kernel[grid](
        q, k, scores, N, D, math.sqrt(D),
        q.stride(0), k.stride(0), scores.stride(0),
        BLOCK_M=TILE, BLOCK_N=TILE, BLOCK_K=TILE, PRECISION=precision,
    )
    return scores


def softmax_rows(scores: torch.Tensor) -> torch.Tensor:
    N = scores.shape[0]
    block = triton.next_power_of_2(N)
    num_warps = 4 if block <= 1024 else 8 if block <= 4096 else 16
    softmax_rows_kernel[(N,)](scores, N, scores.stride(0), BLOCK=block, num_warps=num_warps)
    return scores


def pv(p: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
       precision: str = "ieee") -> torch.Tensor:
    N, D = v.shape
    grid = (triton.cdiv(D, TILE), triton.cdiv(N, TILE))
    pv_kernel[grid](
        p, v, out, N, D,
        p.stride(0), v.stride(0), out.stride(0),
        BLOCK_M=TILE, BLOCK_N=TILE, BLOCK_D=TILE, PRECISION=precision,
    )
    return out


def triton_tiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           scores: torch.Tensor = None, out: torch.Tensor = None,
                           precision: str = "ieee") -> torch.Tensor:
    """Causal single-head attention, 3 kernels. q, k, v: [N, d] float32 CUDA tensors.

    `scores` (N x N) and `out` (N x d) can be passed in so that benchmarks do not
    time allocation, matching the CUDA benchmark, which allocates once up front.
    """
    for x in (q, k, v):
        _check(x)
    N, D = q.shape
    if scores is None:
        scores = torch.empty((N, N), device=q.device, dtype=torch.float32)
    if out is None:
        out = torch.empty((N, D), device=q.device, dtype=torch.float32)
    qkt_scores(q, k, scores, precision)
    softmax_rows(scores)
    pv(scores, v, out, precision)
    return out


# ----------------------------------------------------------------------------
# Quick self-check: python triton_attention/01_triton_tiled_qkt.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'N':>6} {'max |err| vs SDPA (fp32)':>26}")
    for N in (128, 256, 512, 1024):
        q, k, v = (torch.rand(N, 64, device="cuda") * 2 - 1 for _ in range(3))
        out = triton_tiled_attention(q, k, v)
        ref = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None],
                                             is_causal=True)[0, 0]
        print(f"{N:>6} {(out - ref).abs().max().item():>26.3e}")
