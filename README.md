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

# Project Progress

```text
PHASE 1  CPU Self-Attention              ██████████  COMPLETED
PHASE 2  Naive CUDA Implementation       ██████████  COMPLETED
PHASE 3  Shared Memory Optimization      ██████████  COMPLETED
PHASE 4  Performance Benchmarking        ██████████  COMPLETED

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
