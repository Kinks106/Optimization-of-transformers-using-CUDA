"""
02_triton_fused_attention.py - one fused causal attention kernel (FlashAttention-style).

Each program owns one tile of BLOCK_M query rows and loops over K/V tiles:

    for each K/V tile j (only tiles at or left of the diagonal: causal):
        S   = Q_tile @ K_j^T * (1/sqrt(d))       (in registers; never written to memory)
        S   = mask(S, col > row -> -inf)
        m'  = max(m, rowmax(S))                   online softmax:
        P   = exp(S - m')                         keep a running max m and running sum l,
        l   = l * exp(m - m') + rowsum(P)         and rescale the partial output when
        acc = acc * exp(m - m') + P @ V_j         the max changes
        m   = m'
    O_tile = acc / l

Compared with 01 (and with our CUDA code):
  * One kernel launch instead of three.
  * The N x N scores matrix is never stored: memory traffic is O(N*d), not O(N^2).
  * K/V tiles entirely above the diagonal are skipped, so about half the QK^T work is saved.
  * Tile sizes / warps are picked by triton.autotune on the GPU it runs on.

Precision:
  * FP32 (main, comparable with our CUDA code): FP32 inputs, tl.dot with input_precision="ieee".
  * FP16 (separate, NOT comparable with our FP32 CUDA code): FP16 inputs/outputs,
    FP32 accumulation and FP32 softmax. Shown only as "what tensor cores give you".
"""

import math

import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bm in (32, 64, 128):
        for bn in (32, 64):
            for warps in (4, 8):
                for stages in (2, 3):
                    cfgs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                              num_warps=warps, num_stages=stages))
    return cfgs


@triton.autotune(configs=_configs(), key=["N", "HEAD_DIM", "PRECISION"])
@triton.jit
def fused_causal_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    N, sm_scale,
    stride_q, stride_k, stride_v, stride_o,
    HEAD_DIM: tl.constexpr,
    PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Q tile stays in registers for the whole loop
    q = tl.load(q_ptr + offs_m[:, None] * stride_q + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)   # running row max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                 # running row sum
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)        # unnormalized output

    # Causal: keys after this tile's last query row are never needed
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + offs_n[None, :] * stride_k + offs_d[:, None],
                     mask=offs_n[None, :] < N, other=0.0)          # K^T tile [d, BLOCK_N]
        v = tl.load(v_ptr + offs_n[:, None] * stride_v + offs_d[None, :],
                    mask=offs_n[:, None] < N, other=0.0)           # V tile   [BLOCK_N, d]

        s = tl.dot(q, kt, input_precision=PRECISION) * sm_scale
        s = tl.where((offs_n[None, :] > offs_m[:, None]) | (offs_n[None, :] >= N),
                     float("-inf"), s)

        # Online softmax update. Key 0 is in the first tile and is never masked,
        # so m_new is finite for every row from the first iteration on.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)                # rescale factor for the old partial results
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc, input_precision=PRECISION)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * stride_o + offs_d[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < N)


def triton_fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           out: torch.Tensor = None) -> torch.Tensor:
    """Causal single-head attention in one kernel. q, k, v: [N, d] CUDA tensors.

    float32 inputs -> full FP32 (no TF32). float16 inputs -> the separate FP16 variant.
    """
    assert q.is_cuda and q.dim() == 2 and q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.float32, torch.float16)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    N, D = q.shape
    assert D in (16, 32, 64, 128), "head_dim must be a power of two between 16 and 128"
    if out is None:
        out = torch.empty_like(q)
    # FP32: force IEEE FP32 in tl.dot. FP16: inputs are already FP16, so the
    # setting only matters for FP32 operands; we pass the Triton default.
    precision = "ieee" if q.dtype == torch.float32 else "tf32"
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]),)
    fused_causal_attention_kernel[grid](
        q, k, v, out, N, 1.0 / math.sqrt(D),
        q.stride(0), k.stride(0), v.stride(0), out.stride(0),
        HEAD_DIM=D, PRECISION=precision,
    )
    return out


# ----------------------------------------------------------------------------
# Quick self-check: python triton_attention/02_triton_fused_attention.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'N':>6} {'FP32 max |err|':>16} {'FP16 max |err| (separate)':>27}")
    for N in (128, 256, 512, 1024):
        q, k, v = (torch.rand(N, 64, device="cuda") * 2 - 1 for _ in range(3))
        sdpa = lambda a, b, c: F.scaled_dot_product_attention(
            a[None, None], b[None, None], c[None, None], is_causal=True)[0, 0]
        err32 = (triton_fused_attention(q, k, v) - sdpa(q, k, v)).abs().max().item()
        q16, k16, v16 = q.half(), k.half(), v.half()
        err16 = (triton_fused_attention(q16, k16, v16).float() - sdpa(q, k, v)).abs().max().item()
        cfg = fused_causal_attention_kernel.best_config
        print(f"{N:>6} {err32:>16.3e} {err16:>27.3e}   fp16 config: {cfg}")
