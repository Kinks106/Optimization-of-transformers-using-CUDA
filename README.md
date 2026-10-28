# Optimizing Self and Multi-Head Attention Architecture of Transformer Using CUDA

## Project Overview

This project focuses on implementing and optimizing the **self-attention architecture used in Transformers** using **CUDA C++**.

The project is based on the mathematical operations behind the attention mechanism of a GPT/Transformer model. Instead of relying entirely on high-level PyTorch operations, the core attention computations are implemented manually using:

* C++
* CUDA C++
* Custom CUDA kernels
* GPU parallelism
* Shared memory optimization
* Tiled matrix multiplication

The project demonstrates the progression from a sequential CPU implementation to a naive CUDA implementation and finally to an optimized CUDA implementation.

The mathematical attention operation is:

$$
Attention(Q,K,V) =
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
+
Mask
\right)V
$$

---

# Project Objectives

The main objectives of this project are:

1. Understand the mathematical implementation of Transformer self-attention.
2. Implement self-attention from scratch using C++.
3. Parallelize attention computation using CUDA.
4. Optimize the `Q × Kᵀ` operation using shared memory and tiled computation.
5. Compare CPU, naive CUDA, and optimized CUDA implementations.
6. Analyze how CUDA performance changes with increasing sequence length.
7. Extend the implementation toward optimized multi-head attention during the remaining project period.

---

# Current Implementation Status

## Completed

### Phase 1: CPU Self-Attention

A sequential CPU implementation of self-attention was created using standard C++.

The implementation performs:

```text
Q × Kᵀ
   ↓
Scaling by √dₖ
   ↓
Causal Mask
   ↓
Row-wise Softmax
   ↓
Attention × V
```

File:

```text
cuda_attention/01_cpu_attention.cpp
```

---

### Phase 2: Naive CUDA Self-Attention

The attention computation was moved to the GPU using CUDA kernels.

The basic thread mapping is:

```text
Thread(row, col)
        ↓
Computes Attention Score[row][col]
```

The CUDA implementation includes:

* GPU memory allocation
* Host to Device memory transfer
* CUDA kernel execution
* Parallel attention score computation
* Causal masking
* Row-wise softmax
* Attention × V computation
* Device to Host memory transfer

File:

```text
cuda_attention/02_naive_cuda_attention.cu
```

---

### Phase 3: Optimized CUDA Attention

The optimized implementation improves the attention score computation using:

* Shared memory
* Tiled matrix multiplication
* 2D CUDA thread blocks
* Reduced global memory access
* CUDA event timing

Instead of repeatedly loading data from global memory, blocks of `Q` and `K` are loaded into shared memory.

```text
Global Memory
      │
      ▼
┌──────────────────┐
│  Shared Memory   │
│                  │
│  Q Tile          │
│  K Tile          │
└──────────────────┘
      │
      ▼
Multiple CUDA Threads
```

File:

```text
cuda_attention/03_optimized_attention.cu
```

---

### Phase 4: Performance Benchmarking

The project benchmarks:

```text
CPU Sequential
        ↓
Naive CUDA
        ↓
Optimized CUDA
```

The benchmark tests multiple sequence lengths:

```text
128
256
512
1024
```

with:

```text
Head Dimension = 64
Tile Size = 16 × 16
```

File:

```text
cuda_attention/benchmark.cu
```

---

# Project Structure

```text
GPT-from-scratch/
│
├── bigram.py
├── demo.py
├── gpt_dev.ipynb
├── input.txt
├── requirements.txt
├── v2.py
│
├── cuda_attention/
│   │
│   ├── 01_cpu_attention.cpp
│   │   └── Sequential CPU implementation
│   │
│   ├── 02_naive_cuda_attention.cu
│   │   └── Basic CUDA implementation
│   │
│   ├── 03_optimized_attention.cu
│   │   └── Shared memory optimized attention
│   │
│   ├── benchmark.cu
│   │   └── CPU vs CUDA performance benchmark
│   │
│   └── CMakeLists.txt
│
├── triton_attention/              (Content Beyond Syllabus: Triton)
│   ├── SETUP.md                   Windows setup + troubleshooting
│   ├── 00_vector_add.py           Triton smoke test
│   ├── 01_triton_tiled_qkt.py     Triton port of our 3-kernel CUDA algorithm
│   ├── 02_triton_fused_attention.py  Fused causal attention (online softmax)
│   ├── check_correctness.py       Errors vs PyTorch SDPA
│   ├── benchmark_triton.py        Fair benchmark of all GPU versions
│   ├── cuda_ext/                  Our CUDA kernels as a PyTorch extension
│   └── results/                   Generated tables (.md) and raw timings (.json)
│
└── README.md
```

---

# System Requirements

The project was developed and tested using:

```text
Operating System: Windows 11
GPU: NVIDIA GeForce RTX 4060
CUDA Toolkit: CUDA 13.3
CUDA Compiler: nvcc 13.3
C++ Compiler: Microsoft Visual C++ 2022
Architecture: x64
```

An NVIDIA GPU with CUDA support is required.

---

# Installation and Setup

## 1. Install CUDA Toolkit

Install the NVIDIA CUDA Toolkit.

Verify the installation:

```cmd
nvcc --version
```

Expected output should show the installed CUDA version.

Example:

```text
Cuda compilation tools, release 13.3
```

---

## 2. Install Visual Studio Community

Install Visual Studio Community 2022 with:

```text
Desktop development with C++
```

Make sure the following components are installed:

```text
MSVC C++ x64/x86 Build Tools
Windows SDK
```

---

## 3. Open the Visual Studio Developer Environment

Open Command Prompt and initialize the Visual Studio x64 compiler environment:

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
```

Verify the compiler:

```cmd
cl
```

The output should contain:

```text
for x64
```

Verify CUDA:

```cmd
nvcc --version
```

---

# Running the Project

Navigate to the CUDA implementation directory:

```cmd
cd "D:\College\SEM7\GPU\pROJECT\GPT from Scratch\cuda_attention"
```

> Replace the path with the location where the repository exists on your system.

---

# 1. Run CPU Self-Attention

Compile:

```cmd
cl /EHsc /O2 01_cpu_attention.cpp /Fe:cpu_attention.exe
```

Run:

```cmd
cpu_attention.exe
```

This executes the complete self-attention calculation sequentially on the CPU.

---

# 2. Run Naive CUDA Attention

Compile:

```cmd
nvcc -O3 -arch=sm_89 02_naive_cuda_attention.cu -o naive_attention.exe
```

Run:

```cmd
naive_attention.exe
```

This executes the attention calculation using CUDA kernels.

---

# 3. Run Optimized CUDA Attention

Compile:

```cmd
nvcc -O3 -arch=sm_89 03_optimized_attention.cu -o optimized_attention.exe
```

Run:

```cmd
optimized_attention.exe
```

The implementation uses shared memory and tiled matrix multiplication.

---

# 4. Run Benchmark

Compile:

```cmd
nvcc -O3 -arch=sm_89 benchmark.cu -o benchmark.exe
```

Run:

```cmd
benchmark.exe
```

The benchmark compares:

```text
CPU Sequential
Naive CUDA
Optimized CUDA
```

---

# Benchmark Results

The current implementation was tested with the following configuration:

```text
GPU: NVIDIA GeForce RTX 4060
Head Dimension: 64
CUDA Iterations: 20
```

| Sequence Length | CPU (ms) | Naive CUDA (ms) | Optimized CUDA (ms) | Speedup |
| --------------: | -------: | --------------: | ------------------: | ------: |
|             128 |   0.3259 |          0.0556 |              0.0517 |   6.30× |
|             256 |   1.5480 |          0.0671 |              0.0481 |  32.16× |
|             512 |   7.0986 |          0.1849 |              0.1315 |  53.97× |
|            1024 |  30.7340 |          0.6447 |              0.4365 |  70.41× |

The speedup is calculated as:

$$
Speedup =
\frac{CPU\ Execution\ Time}
{Optimized\ CUDA\ Execution\ Time}
$$

The highest measured speedup in the current benchmark is:

```text
70.41×
```

for sequence length:

```text
1024
```

---

# Performance Analysis

The results show that CUDA acceleration becomes more beneficial as sequence length increases.

```text
Sequence Length     Speedup
────────────────────────────
128                 6.30×
256                 32.16×
512                 53.97×
1024                70.41×
```

Self-attention has approximately quadratic complexity with respect to sequence length:

$$
O(n^2 \cdot d)
$$

As the sequence length increases, the number of attention scores increases significantly.

CUDA allows many of these calculations to execute in parallel.

The optimized implementation further improves performance through:

```text
Shared Memory
      +
Tiled Computation
      +
Parallel Thread Execution
```

---

# CUDA Optimization Strategy

## Naive CUDA

Each thread directly accesses global memory.

```text
Thread
  │
  ├── Read Q from Global Memory
  ├── Read K from Global Memory
  └── Calculate one attention score
```

This causes repeated global memory accesses.

---

## Optimized CUDA

The optimized implementation uses shared memory.

```text
Global Memory
       │
       ▼
 Load Q and K Tiles
       │
       ▼
 Shared Memory
       │
       ▼
 Multiple Threads Reuse Data
       │
       ▼
 Attention Scores
```

This reduces repeated global memory accesses and improves memory locality.

---

# Current Limitations

The current project focuses primarily on optimizing the `Q × Kᵀ` computation.

The following operations are currently functional but not fully optimized:

```text
Row-wise Softmax
Attention × V
```

The current softmax implementation uses one active thread per attention row.

This creates an opportunity for further CUDA optimization.

---

# Future Work

The project is planned to continue during the remaining half-semester.

## Phase 5: Parallel Softmax

Implement:

* Parallel maximum reduction
* Parallel exponential calculation
* Parallel sum reduction
* Shared-memory reduction

Current:

```text
One Row
   ↓
One Active Thread
```

Planned:

```text
One Row
   ↓
One CUDA Block
   ↓
Multiple Threads
   ↓
Parallel Reduction
```

---

## Phase 6: Optimize Attention × V

Implement tiled shared-memory multiplication for:

$$
Softmax(QK^T)V
$$

This will optimize the second major matrix multiplication in the attention mechanism.

---

## Phase 7: Multi-Head Attention

Extend the current single-head attention implementation to multi-head attention.

```text
Input
  │
  ├── Head 0
  ├── Head 1
  ├── Head 2
  └── Head 3
        │
        ▼
Parallel CUDA Execution
        │
        ▼
Concatenation
```

Different attention heads can be processed independently on the GPU.

---

## Phase 8: Kernel Fusion

Investigate combining compatible operations to reduce:

```text
Kernel Launch Overhead
+
Intermediate Global Memory Access
```

---

## Phase 9: GPT Integration

Integrate the custom CUDA attention implementation with the existing GPT-from-scratch architecture.

The original repository provides the Transformer/GPT context, while this project replaces the core attention computation with custom CUDA kernels.

---

## Phase 10: GPU Profiling

Future profiling will include:

* CUDA profiling tools
* Kernel execution analysis
* Memory access analysis
* Occupancy analysis
* Bottleneck identification

---

# Content Beyond Syllabus: Triton

As an extension beyond the course syllabus, the same causal single-head attention was
implemented in **OpenAI Triton**. It was then benchmarked against our own CUDA kernels
using one shared timing method.

All code is in `triton_attention/`. The original files in `cuda_attention/` were not modified.

## What is Triton?

Triton is a Python-embedded language and compiler for writing GPU kernels. A kernel is a
Python function decorated with `@triton.jit`. Triton compiles it at run time
(just-in-time) to PTX, the same GPU assembly that `nvcc` produces, and then to machine code for
the GPU it runs on. Triton is used inside PyTorch: `torch.compile` generates Triton
kernels.

## Block-level model (Triton) vs thread-level model (CUDA)

| | CUDA (our kernels) | Triton |
|---|---|---|
| Unit you program | **one thread** (`threadIdx`, `blockIdx`) | **one program** working on a whole **block (tile) of data** (`tl.program_id`) |
| Data you handle | scalars: `Q[row * d + k]` | tensors: `tl.load` of a whole `[16, 16]` tile |
| Shared memory | manual: `__shared__ float Q_tile[16][16]`, `__syncthreads()` | automatic: the compiler decides what goes in shared memory |
| Matrix multiply | hand-written loop over the tile | `tl.dot(a, b)`, which can use tensor cores |
| Bounds checks | `if (row < seq_len && col < seq_len)` per thread | `mask=` argument on loads and stores |
| Memory coalescing, vectorized loads, software pipelining | by hand | done by the compiler (`num_stages`, `num_warps`) |
| Tuning tile sizes | edit constants and recompile | `@triton.autotune` tries several configs on the actual GPU |

The same Q×Kᵀ tile, written both ways (simplified; bounds checks omitted):

```cpp
// CUDA (benchmark.cu): each THREAD computes one score
__shared__ float Q_tile[TILE][TILE];
__shared__ float K_tile[TILE][TILE];
for (int tileStart = 0; tileStart < head_dim; tileStart += TILE) {
    Q_tile[threadIdx.y][threadIdx.x] = Q[row * head_dim + tileStart + threadIdx.x];
    K_tile[threadIdx.y][threadIdx.x] = K[col * head_dim + tileStart + threadIdx.y];
    __syncthreads();
    for (int k = 0; k < TILE; k++) sum += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];
    __syncthreads();
}
```

```python
# Triton (01_triton_tiled_qkt.py): each PROGRAM computes a whole 16x16 tile
for k0 in range(0, D, BLOCK_K):
    ks = k0 + tl.arange(0, BLOCK_K)
    q  = tl.load(q_ptr + rows[:, None] * stride_q + ks[None, :], mask=..., other=0.0)
    kt = tl.load(k_ptr + cols[None, :] * stride_k + ks[:, None], mask=..., other=0.0)
    acc = tl.dot(q, kt, acc, input_precision="ieee")   # full FP32, no TF32
```

## Implementations

| File | What it is |
|---|---|
| `01_triton_tiled_qkt.py` | **Algorithm-for-algorithm port** of our optimized CUDA version: 16×16 tiles for Q×Kᵀ, the full N×N score matrix in global memory, then separate softmax and P×V kernels (3 kernels). |
| `02_triton_fused_attention.py` | **One fused kernel** in the FlashAttention style: each program loads a tile of Q, loops over K/V tiles with an *online softmax* (running max and sum), and skips K/V tiles above the causal diagonal. The N×N matrix is never stored. Tile sizes are autotuned. |
| `cuda_ext/` | Our naive and optimized CUDA kernels, **copied verbatim from `benchmark.cu`** and built as a PyTorch extension, so they can be timed with the same Python timer. The benchmark checks that the kernel text still matches `benchmark.cu`. |

**Precision:** our CUDA code is FP32, so all main results are FP32. On Ampere/Ada GPUs, `tl.dot`
uses TF32 by default, which is less precise, so we force `input_precision="ieee"`. The smoke test
shows why this matters: a 64×64 FP32 product has error 2.7e-6 with `ieee` and 7.3e-3 with TF32.
An FP16 variant of the fused kernel is reported **separately**, because it is not comparable with
FP32 CUDA.

## Setup

Native Windows, using the community `triton-windows` package (full guide and troubleshooting:
[`triton_attention/SETUP.md`](triton_attention/SETUP.md)):

```cmd
py -3.13 -m venv .venv-triton
.venv-triton\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv-triton\Scripts\python.exe -m pip install "triton-windows>=3.8,<3.9" ninja numpy
.venv-triton\Scripts\python.exe triton_attention\00_vector_add.py
```

* A separate venv is used because `requirements.txt` installs a CUDA 12.1 build of torch.
  `torch.utils.cpp_extension` needs torch's CUDA **major** version to match `nvcc` (13.3).
* `triton-windows` is an unofficial fork of Triton for Windows. It bundles `ptxas` and a small C
  compiler, so Triton kernels run without MSVC or a CUDA toolkit. Building our CUDA kernels as an
  extension still needs both (run from the *x64 Native Tools Command Prompt for VS 2022*).
* Windows problem found: the 260-character path limit. When the venv path was long, Triton
  failed to import (`DLL load failed ... filename or extension is too long`), and `cl.exe` and `nvcc` could
  not find headers that existed (`Cannot open include file: 'detail/class.h'`). A short venv path fixed both.

Run:

```cmd
.venv-triton\Scripts\python.exe triton_attention\check_correctness.py
.venv-triton\Scripts\python.exe triton_attention\benchmark_triton.py
```

## Benchmark method

Every GPU version is timed from the same Python process in the same way:

* same Q, K, V tensors (independent U[-1, 1], fixed seed), d = 64, already on the GPU
* output and score buffers allocated once, outside the timed region (as in `benchmark.cu`)
* every version is checked against PyTorch SDPA before it is timed
* CUDA events around 100 back-to-back calls, after 10 warm-up calls, repeated 5 rounds,
  **median round reported** (all rounds are saved in `results/*.json`)
* two modes: **CUDA graph replay** (GPU time only) and **eager** (normal Python calls,
  which include CPU launch overhead)
* no L2 cache flush between calls (as in `benchmark.cu`)

## Correctness

Max absolute error against `torch.nn.functional.scaled_dot_product_attention(is_causal=True)` (FP32), d = 64:

| Seq len | naive CUDA | optimized CUDA | Triton tiled | Triton fused | Triton fused FP16 (separate) |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.24e-07 | 2.24e-07 | 2.98e-07 | 2.38e-07 | 3.78e-04 |
| 256 | 2.24e-07 | 2.24e-07 | 2.09e-07 | 2.09e-07 | 3.57e-04 |
| 512 | 1.53e-07 | 1.53e-07 | 2.09e-07 | 1.79e-07 | 4.14e-04 |
| 1024 | 1.94e-07 | 1.94e-07 | 1.79e-07 | 1.79e-07 | 3.59e-04 |

All FP32 versions agree with SDPA to rounding level (< 3e-7). Measured against a float64
reference, the Triton FP32 results are as accurate as SDPA's own FP32 output (see
`triton_attention/results/correctness_*.md`).

## Results

### RTX 4060 (project machine)

> **To be filled in:** run `benchmark_triton.py` on the RTX 4060 and copy the tables from
> `triton_attention/results/benchmark_nvidia_geforce_rtx_4060*.md`.
> The numbers below are from the development laptop and should not be quoted as RTX 4060 results.
> The earlier RTX 4060 table in [Benchmark Results](#benchmark-results) was measured by `benchmark.cu`
> with a different timer (20 iterations, no CUDA graphs), so it should not be mixed with these tables.

| Seq len | naive CUDA | optimized CUDA | Triton tiled | Triton fused | PyTorch SDPA |
|---:|---:|---:|---:|---:|---:|
| 128 | — | — | — | — | — |
| 256 | — | — | — | — | — |
| 512 | — | — | — | — | — |
| 1024 | — | — | — | — | — |

### RTX 3050 6GB Laptop GPU (development machine, sm_86)

torch 2.14.1+cu130, triton-windows 3.8.0, FP32, d = 64, time per attention call in ms.

**CUDA graph replay (GPU time only):**

| Seq len | naive CUDA | optimized CUDA | Triton tiled | Triton fused | PyTorch SDPA |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0340 | 0.0239 | 0.0105 | 0.0156 | 0.0263 |
| 256 | 0.0780 | 0.0545 | 0.0238 | 0.0276 | 0.0471 |
| 512 | 0.2506 | 0.1767 | 0.0718 | 0.0535 | 0.0895 |
| 1024 | 0.9193 | 0.6422 | 0.3142 | 0.1422 | 0.1768 |

Speedup over **optimized CUDA** (graph mode; >1 means faster):

| Seq len | naive CUDA | Triton tiled | Triton fused | PyTorch SDPA |
|---:|---:|---:|---:|---:|
| 128 | 0.70× | 2.26× | 1.53× | 0.91× |
| 256 | 0.70× | 2.29× | 1.97× | 1.16× |
| 512 | 0.71× | 2.46× | 3.31× | 1.97× |
| 1024 | 0.70× | 2.04× | 4.52× | 3.63× |

**Eager (normal Python calls, includes launch overhead):**

| Seq len | naive CUDA | optimized CUDA | Triton tiled | Triton fused | PyTorch SDPA |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0369 | 0.0333 | 0.0587 | 0.0405 | 0.0332 |
| 256 | 0.0811 | 0.0591 | 0.0614 | 0.0319 | 0.0548 |
| 512 | 0.2538 | 0.1797 | 0.0759 | 0.0542 | 0.0930 |
| 1024 | 0.9217 | 0.6424 | 0.3164 | 0.1433 | 0.1804 |

**Per-stage time at N = 1024 (graph mode, ms):**

| Stage | naive CUDA | optimized CUDA | Triton tiled |
|---|---:|---:|---:|
| 1. Q×Kᵀ + scale + mask | 0.5048 | 0.2201 | 0.2017 |
| 2. softmax | 0.2852 | 0.2852 | 0.0546 |
| 3. P×V | 0.1422 | 0.1422 | 0.0597 |

(naive and optimized CUDA share the same softmax and P×V kernels.)

**FP16, separate (graph mode, ms):** Triton fused FP16: 0.0078 / 0.0169 / 0.0138 / 0.0256;
PyTorch SDPA FP16: 0.0120 / 0.0167 / 0.0288 / 0.0530 (N = 128 / 256 / 512 / 1024).

**Run-to-run variation:** in graph mode the slowest and fastest rounds differ by 5% (N = 1024) to 38% (N = 256).
In eager mode they differ by up to about 100%, because laptop GPU clocks change and small kernels are CPU-bound.
Medians are reported; differences of a few percent are within noise.

## Lines of kernel code

Device code only, excluding blank and comment lines, counted automatically by `benchmark_triton.py`:

| Implementation | Kernel lines | Kernels | Hand-managed shared memory |
|---|---:|---:|:---:|
| naive CUDA | 69 | 3 | no |
| optimized CUDA | 91 | 3 | yes |
| Triton tiled | 50 | 3 | no (compiler) |
| Triton fused | 36 | 1 | no (compiler) |
| PyTorch SDPA | — (library call) | — | — |

## Analysis: performance vs development effort

These observations are from the RTX 3050 measurements above. They should be rechecked against the RTX 4060 tables.

1. **The same algorithm gives about the same speed.** For the one stage written the same way in both
   languages (tiled Q×Kᵀ, 16×16 tiles), Triton and our shared-memory CUDA kernel are within
   about 9% (0.2017 vs 0.2201 ms at N = 1024; 0.0514 vs 0.0556 ms at N = 512). The Triton version needs
   no `__shared__` arrays, no `__syncthreads()` and no per-thread index arithmetic.

2. **Most of Triton tiled's 2.0–2.5× advantage comes from the stages we have not optimized yet.**
   Our softmax uses one thread per row, serially (0.2852 ms at N = 1024). A Triton program
   naturally processes the whole row as a vector, so the same 3-pass softmax takes 0.0546 ms (5.2×).
   Our P×V kernel does not use shared memory. Triton's `tl.dot` version is 2.4× faster. These are
   exactly the planned Phases 5 and 6 in CUDA. In Triton they came almost for free from the block model.

3. **Fusion is the largest win, and it is short in Triton.** The fused kernel (36 kernel lines)
   is 4.5× faster than our optimized CUDA at N = 1024, and faster than PyTorch's FP32 SDPA at every
   tested length in graph mode (for example 0.1422 vs 0.1768 ms at N = 1024). It never writes
   the N×N matrix and skips the masked half of the work. Doing the same in CUDA (our planned
   Phase 8) would mean hand-writing online softmax, shared-memory tiling for both matmuls and the tile skipping.

4. **The fused kernel is not the fastest at small N.** At N = 128 and 256, Triton tiled (0.0105 and
   0.0238 ms) beats Triton fused (0.0156 and 0.0276 ms). The likely reason: with a single head and no batch, the
   fused kernel launches only N / BLOCK_M programs. The autotuner picked BLOCK_M = 32, which gives
   4 programs at N = 128 and 8 at N = 256: too few to fill the RTX 3050's 20 SMs. Real models run many heads and batches in parallel, which removes this limit.

5. **Launch overhead matters at small sizes.** In eager mode at N = 128, all versions take 0.03–0.06 ms,
   because launching kernels from Python costs more than the GPU work itself. Triton tiled launches 3 kernels from
   Python, so it is slower than optimized CUDA there (0.0587 vs 0.0333 ms). CUDA graphs remove this overhead.

6. **Development effort.** The Triton kernels are shorter (50 and 36 lines against 69 and 91) and need no
   manual shared-memory management. Autotuning picked tile sizes per GPU with no code changes.
   The costs were: setup on Windows depends on a community fork; the 260-character path limit broke
   both the Triton import and the extension build until the venv was moved; the first call of each kernel pays a
   JIT compile (and autotuning) cost; and the default TF32 behaviour of `tl.dot` silently
   lowers FP32 accuracy unless `input_precision="ieee"` is set. CUDA gives full control over threads and memory, but
   every optimization (parallel reductions, tiling, fusion) has to be written by hand.

**Summary:** for the same algorithm, Triton matches hand-written CUDA with less code. Its bigger
benefit is that the block-level model makes the next optimizations (parallel softmax, tiled
P×V, fusion) short and simple to write. On the development GPU, that took attention from
0.6422 ms (our optimized CUDA) to 0.1422 ms (Triton fused) at N = 1024 in FP32.

---

# Project Progress

```text
PHASE 1  CPU Self-Attention              ██████████  COMPLETED
PHASE 2  Naive CUDA Implementation       ██████████  COMPLETED
PHASE 3  Shared Memory Optimization      ██████████  COMPLETED
PHASE 4  Performance Benchmarking        ██████████  COMPLETED
BEYOND   Triton Comparison               ██████████  COMPLETED

PHASE 5  Parallel Softmax                ░░░░░░░░░░  PLANNED
PHASE 6  Optimized Attention × V         ░░░░░░░░░░  PLANNED
PHASE 7  Multi-Head Attention            ░░░░░░░░░░  PLANNED
PHASE 8  Kernel Fusion                   ░░░░░░░░░░  PLANNED
PHASE 9  GPT Integration                 ░░░░░░░░░░  PLANNED
PHASE 10 GPU Profiling                   ░░░░░░░░░░  PLANNED
```

---

# Conclusion

This project demonstrates the implementation and optimization of Transformer self-attention using CUDA.

Starting from a sequential CPU implementation, the attention mechanism was progressively transformed into:

```text
CPU Sequential Implementation
              ↓
Naive CUDA Parallelization
              ↓
Shared Memory Optimized CUDA
              ↓
Performance Benchmarking
```

The current implementation achieved up to:

# 70.41× Speedup

over the CPU baseline for the tested configuration.

The project will continue by optimizing the remaining attention operations and extending the implementation to multi-head attention and the full GPT architecture.
