# Optimizing the Self- and Multi-Head Attention Architecture of Transformers using CUDA

### A GPU Computing Project Report

**Course:** GPU Computing (Semester 7)
**Repository:** `Optimization-of-transformers-using-CUDA` — branch `feature/triton-attention`
**Target hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (compute capability `sm_89`), CUDA 13.3, Windows 11, MSVC 2022

---

## Abstract

Self-attention is the computational core of every Transformer and GPT model, and it is
also its most expensive operation: its cost grows **quadratically** with the length of
the input sequence. This project implements the self-attention mechanism *from scratch*
and then accelerates it on the GPU, measuring the speed-up at every step. We progress
from a sequential C++ implementation on the CPU, to a naive CUDA implementation that
assigns one GPU thread per attention score, to an optimized CUDA implementation that
uses **shared memory** and **tiled matrix multiplication** to reduce slow global-memory
traffic. On the project GPU (RTX 4060) the optimized CUDA kernel reaches roughly a
**70× speed-up** over the CPU baseline at a sequence length of 1024.

We then integrate these custom kernels back into a small "GPT from scratch" language
model, so that the GPT's attention head can run on our own CUDA code, and we verify that
the custom kernels reproduce the reference PyTorch output. A Streamlit user interface
showcases the benchmark results and the live properties of the GPU the kernels run on.

The report has two parts. **Part 1** covers the entire project: the mathematics, the
GPU background, the three implementations, the benchmark methodology and results, the
GPT integration and the UI. **Part 2** covers the *content beyond syllabus*: the same
attention written in **OpenAI Triton**, including a FlashAttention-style fused kernel,
and a fair, like-for-like comparison of hand-written CUDA against Triton and PyTorch's
own optimized attention — along with an analysis of performance versus development
effort.

---

## Table of Contents

**Part 1 — The Project**
1. Introduction
2. Background: Transformers, attention, and the GPU
3. The GPT-from-scratch context
4. Phase 1 — CPU self-attention
5. Phase 2 — Naive CUDA self-attention
6. Phase 3 — Optimized CUDA (shared memory + tiling)
7. Phase 4 — Benchmarking methodology and results
8. Phase 9 — GPT integration
9. The showcase UI and GPU properties
10. Project structure and build
11. Limitations and future work

**Part 2 — Content Beyond Syllabus: Triton**
12. What Triton is and why we used it
13. Block-level vs thread-level programming
14. Implementation A — Triton tiled (3-kernel port)
15. Implementation B — Triton fused (online softmax)
16. Numerical precision: TF32, IEEE FP32, FP16
17. Fair benchmarking methodology
18. Correctness results
19. Performance results
20. Per-stage and lines-of-code analysis
21. Discussion: performance vs development effort
22. Windows setup challenges

**Back matter** — Conclusion, References, Appendix

---
---

# PART 1 — THE PROJECT

---

## 1. Introduction

### 1.1 Motivation

Transformer models — the architecture behind GPT, BERT and essentially all modern large
language models — spend the majority of their compute inside the **self-attention**
operation. For an input of *n* tokens, attention compares *every* token with *every*
other token, producing an *n × n* matrix of interaction scores. The amount of work is
therefore proportional to *n²*, which means that doubling the sequence length roughly
**quadruples** the cost. For long sequences this single operation dominates both the
runtime and the memory use of the whole model.

Because the *n²* score computations are largely independent of one another, attention is
an ideal candidate for the **massively parallel** execution that a GPU provides. A modern
GPU contains thousands of arithmetic units that can run thousands of threads at the same
time; if we can map the independent pieces of attention onto those threads, we can
compute in parallel what the CPU must compute one step at a time.

This project takes that idea and makes it concrete and measurable. Rather than relying on
a high-level library such as PyTorch to perform attention for us, we implement the
mechanism ourselves — first on the CPU, then on the GPU — so that we can *see* exactly
where the parallelism comes from and *measure* how much faster each version runs.

### 1.2 Objectives

1. Understand the mathematical definition of Transformer self-attention.
2. Implement self-attention from scratch in C++ (a correct, readable baseline).
3. Parallelize the computation using CUDA.
4. Optimize the dominant `Q × Kᵀ` operation using shared memory and tiling.
5. Benchmark CPU, naive CUDA and optimized CUDA across several sequence lengths.
6. Analyze how the speed-up changes as the sequence length grows.
7. Integrate the custom CUDA attention back into the GPT model.
8. (Beyond syllabus) Re-implement the same attention in OpenAI Triton and compare.

### 1.3 Contributions

- A clear, staged implementation of causal self-attention: **CPU → naive CUDA →
  optimized CUDA**, each kept as a separate, runnable program.
- A fair benchmark harness that measures all three on identical inputs.
- A **PyTorch extension** that exposes the CUDA kernels to Python, and a GPT whose
  attention backend can be switched between PyTorch and CUDA, with an automatic
  correctness check.
- A **Streamlit UI** that displays the results and the live GPU properties.
- A beyond-syllabus **Triton** study, including a fused FlashAttention-style kernel and a
  rigorous comparison against hand-written CUDA and PyTorch's own attention.

---

## 2. Background: Transformers, attention, and the GPU

### 2.1 The attention operation

Each token in a sequence is represented by three learned vectors:

- **Query (Q)** — "what is this token looking for?"
- **Key (K)** — "what does this token offer?"
- **Value (V)** — "what information does this token carry forward?"

Self-attention is defined as:

$$
\text{Attention}(Q,K,V) = \text{Softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}} + M\right)V
$$

where *dₖ* is the dimension of each key/query vector (the *head dimension*) and *M* is a
mask. The operation can be read as four steps:

| Step | Formula | Meaning |
|------|---------|---------|
| 1 | `S = Q·Kᵀ / √dₖ` | Similarity of every query to every key (an *n × n* matrix). The division by √dₖ keeps the numbers in a stable range. |
| 2 | `S = S + M` | Add the mask. For a **causal** (GPT-style) model the mask sets all "future" positions to −∞, so a token can only attend to itself and earlier tokens. |
| 3 | `P = softmax(S)` | Turn each row of scores into a probability distribution that sums to 1. |
| 4 | `O = P·V` | Each output row is a weighted average of the value vectors, using the attention probabilities as weights. |

### 2.2 Why the causal mask matters

A GPT predicts the next token from the tokens before it. If token *i* were allowed to look
at token *j > i*, the model could "cheat" by peeking at the answer during training. The
causal mask enforces that each position only attends to positions `0..i`. In our code this
is implemented directly: whenever the column index exceeds the row index, the score is set
to −∞ (which becomes 0 after the softmax).

### 2.3 Computational complexity

Step 1 produces an *n × n* matrix, and each of its *n²* entries is a dot product of length
*dₖ*. Step 4 is another *n × n × dₖ* matrix multiplication. The overall cost is:

$$
O(n^{2}\cdot d_k)
$$

The *n²* term is what makes long sequences expensive, and it is also what makes attention
so suitable for the GPU: there are *n²* nearly-independent score computations, and the GPU
can work on many of them simultaneously.

### 2.4 A short GPU primer

A GPU is built for throughput, not latency. Its execution model has three levels:

- **Thread** — the smallest unit of work; it runs the kernel code for one data element.
- **Block** — a group of threads (in this project, 16×16 = 256 threads) that can
  cooperate: they share a small, fast **shared memory** and can synchronize with
  `__syncthreads()`.
- **Grid** — all the blocks launched for one kernel, together covering the whole problem.

The memory hierarchy is central to performance:

```
       registers      (per-thread, fastest, tiny)
          │
      shared memory    (per-block, very fast, small: a programmer-managed scratchpad)
          │
     global memory     (whole GPU, large, but relatively SLOW — hundreds of cycles)
```

The single most important optimization in this project is to move repeatedly-used data
from slow **global memory** into fast **shared memory** so that many threads can reuse it.

### 2.5 The CUDA programming model

In CUDA C++, a function that runs on the GPU is marked `__global__` (a *kernel*). It is
launched with an execution configuration `kernel<<<grid, block>>>(args)`. Inside the
kernel, each thread finds its own coordinates from the built-in variables `threadIdx`,
`blockIdx` and `blockDim`, and uses them to decide which element of the data it is
responsible for. Data must be explicitly copied between host (CPU) memory and device (GPU)
memory with `cudaMemcpy`, and device memory is allocated with `cudaMalloc`.

### 2.6 Warps, occupancy and latency hiding

Threads do not execute entirely independently. The hardware groups them into **warps** of
32 threads that execute the *same instruction* at the same time (Single-Instruction,
Multiple-Thread, or SIMT). Two consequences matter for this project:

- **Divergence.** If threads in the same warp take different branches of an `if`, the warp
  must execute both paths with some threads disabled. Our causal-mask check (`if (col > row)`)
  causes some divergence along the diagonal, but it is cheap compared with the arithmetic.
- **Latency hiding.** A single warp waiting on a slow global-memory read does not stall the
  GPU; the scheduler simply runs another ready warp. The more warps that are resident on a
  streaming multiprocessor (SM), the better the GPU can hide memory latency. The fraction of
  the maximum possible resident warps that are actually active is called **occupancy**.

A 16×16 block contains 256 threads = 8 warps. With several blocks resident per SM, there are
enough warps to keep the SM busy while others wait on memory — which is why the GPU can
tolerate the naive kernel's heavy memory traffic and still beat the CPU, and why the
optimized kernel (which needs far less memory traffic) runs faster still.

### 2.7 Memory coalescing and why tiling helps

When the 32 threads of a warp read global memory, the hardware is most efficient if they
read **consecutive** addresses: the GPU then combines those reads into one wide memory
transaction (a *coalesced* access). Scattered reads are serviced as many small
transactions and waste most of the memory bandwidth.

Tiling improves this in two complementary ways. First, the cooperative load of a tile is
arranged so that consecutive threads read consecutive addresses, which is coalesced.
Second, and more importantly, once a tile sits in shared memory it is read many times from
there instead of from global memory, so the expensive global accesses happen far less
often. The combination of **fewer** global accesses and **coalesced** global accesses is
what makes the optimized kernel faster.

---

## 3. The GPT-from-scratch context

The repository contains a small character-level GPT (in `gpt/v2.py`), built in the style
of Andrej Karpathy's "Let's build GPT" lecture. It is trained on a plain text file
(`gpt/input.txt`) and learns to generate text one character at a time. Its purpose in this
project is to provide the *context* for attention — to show where, inside a real model,
the operation we are optimizing actually lives.

The model is a stack of Transformer **blocks**, each of which performs:

```
x = x + MultiHeadAttention(LayerNorm(x))   # "communication" between tokens
x = x + FeedForward(LayerNorm(x))          # "computation" within each token
```

Inside one attention head the four lines that matter are exactly our four steps:

```python
wei = q @ k.transpose(-2, -1) * head_size**-0.5   # step 1: Q·Kᵀ / √d
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))  # step 2: causal mask
wei = F.softmax(wei, dim=-1)                       # step 3: softmax
out = wei @ v                                      # step 4: ·V
```

The model uses **multi-head** attention: instead of one attention computation over the
full embedding, it runs several smaller attention "heads" in parallel and concatenates
their outputs. Each head is an independent instance of the same four-step operation, which
is why optimizing a single head is the foundation for optimizing the whole model.

The default configuration in `v2.py` (batch 64, block size 256, embedding 384, 6 heads,
6 layers) produces a model of roughly ten million parameters — small by modern standards,
but structurally identical to a full GPT.

---

## 4. Phase 1 — CPU self-attention

**File:** `cuda_attention/01_cpu_attention.cpp`

The CPU version is the reference implementation. It uses a tiny example (4 tokens, head
dimension 3) with fixed Q, K and V matrices so that every intermediate result can be
inspected and checked by hand. It implements the four steps as four straightforward
functions built from nested `for` loops:

1. **`calculateAttentionScores`** — a triple loop computing `S[i][j] = (Qᵢ · Kⱼ)/√d`.
2. **`applyCausalMask`** — sets `S[i][j] = −∞` whenever `j > i`.
3. **`softmax`** — for each row, subtracts the row maximum (for numerical stability),
   exponentiates, and normalizes so the row sums to 1. Masked (−∞) entries become 0.
4. **`calculateOutput`** — computes `O[i][d] = Σⱼ P[i][j]·V[j][d]`.

A representative excerpt (step 1):

```cpp
for (int i = 0; i < SEQ_LEN; i++)
    for (int j = 0; j < SEQ_LEN; j++) {
        float sum = 0.0f;
        for (int k = 0; k < HEAD_DIM; k++)
            sum += Q[i*HEAD_DIM + k] * K[j*HEAD_DIM + k];   // dot product Qᵢ·Kⱼ
        scores[i*SEQ_LEN + j] = sum / sqrt((float)HEAD_DIM);
    }
```

Everything here is **sequential**: the outer loops run one `(i, j)` pair after another on a
single CPU core. This is correct and easy to read, but it does *n²·d* multiply-adds one at
a time — exactly the work we want to parallelize. The output for the 4×3 example (produced
by running the program) shows the characteristic lower-triangular attention-weight matrix
caused by the causal mask: row 0 attends only to token 0, row 1 splits its attention
between tokens 0 and 1, and so on.

### 4.1 A worked example

With the program's fixed inputs (4 tokens, head dimension 3), the scaled score matrix
`Q·Kᵀ/√3` comes out as:

```
0.4215  0.5196  0.5889  0.4850
0.4157  0.3291  0.4965  0.4330
0.5889  0.5774  0.3984  0.5658
0.6178  0.4099  0.3984  0.5600
```

After the causal mask and row-wise softmax, the attention weights become lower-triangular
(each row sums to 1, upper-triangle zeroed):

```
1.0000  0.0000  0.0000  0.0000
0.5216  0.4784  0.0000  0.0000
0.3552  0.3512  0.2936  0.0000
0.2810  0.2282  0.2256  0.2652
```

Reading row 1: token 1 places 52% of its attention on token 0 and 48% on itself, and none
on the future tokens 2 and 3. Multiplying these weights by `V` gives the final output,
where each row is a blend of the value vectors of the tokens it is allowed to see. This
tiny example is valuable precisely because every number can be verified by hand, giving a
ground truth that the CUDA versions must reproduce.

### 4.2 Numerical stability of the softmax

The softmax subtracts the row maximum before exponentiating (`exp(sᵢ − max)` rather than
`exp(sᵢ)`). Mathematically this changes nothing — the normalization cancels the shift — but
it prevents `exp` from overflowing for large scores. Masked entries (−∞) map to `exp(−∞) = 0`,
so they contribute nothing to the row sum. Every implementation in this project uses this
"safe softmax", and the Triton fused kernel extends the same idea into an *online* form
(Part 2).

---

## 5. Phase 2 — Naive CUDA self-attention

**File:** `cuda_attention/02_naive_cuda_attention.cu`

The first GPU version keeps the same algorithm but assigns **one thread to one output
element**. For the score matrix, a 2-D grid of threads is launched so that the thread at
position `(row, col)` computes `S[row][col]`:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < seq_len && col < seq_len) {
    if (col > row) { scores[row*seq_len + col] = -INFINITY; return; }  // causal mask
    float sum = 0.0f;
    for (int k = 0; k < head_dim; k++)
        sum += Q[row*head_dim + k] * K[col*head_dim + k];
    scores[row*seq_len + col] = sum / sqrtf((float)head_dim);
}
```

Now the *n²* score computations happen (largely) at the same time instead of one after the
other, which is the source of the first large speed-up. The CUDA program also performs the
full pipeline: it allocates device memory, copies Q/K/V from host to device, launches the
score kernel, a softmax kernel and an output kernel, then copies the result back.

**The weakness of the naive version** is memory traffic. Every thread reads its entire row
of Q and its entire column of K directly from **global memory**. Neighbouring threads
re-read many of the same values — the thread for `(row, 0)` and the thread for `(row, 1)`
both read all of row `row` of Q from global memory independently. Because global memory is
slow, these repeated accesses become the bottleneck.

---

## 6. Phase 3 — Optimized CUDA (shared memory + tiling)

**File:** `cuda_attention/03_optimized_attention.cu` (and the same kernels in `benchmark.cu`)

The optimized version attacks the memory-traffic problem with **tiling**. Instead of each
thread fetching from global memory on its own, a block of 16×16 threads cooperatively loads
a 16×16 **tile** of Q and a 16×16 tile of K into shared memory, and then every thread in the
block computes its partial dot product from that fast shared copy. The block marches across
the head dimension one tile at a time, accumulating the result:

```cpp
__shared__ float Q_tile[TILE][TILE];
__shared__ float K_tile[TILE][TILE];
float sum = 0.0f;
for (int tileStart = 0; tileStart < head_dim; tileStart += TILE) {
    Q_tile[threadIdx.y][threadIdx.x] = Q[row*head_dim + tileStart + threadIdx.x];
    K_tile[threadIdx.y][threadIdx.x] = K[col*head_dim + tileStart + threadIdx.y];
    __syncthreads();                       // wait until the whole tile is loaded
    for (int k = 0; k < TILE; k++)
        sum += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];
    __syncthreads();                       // wait before overwriting the tile
}
```

The key ideas:

- **Data reuse.** Each value loaded into shared memory is used by `TILE` threads instead of
  being re-fetched from global memory by each of them. For a 16×16 tile this cuts
  global-memory reads for the Q·Kᵀ step by roughly a factor of 16.
- **Coalesced loads.** Threads load *consecutive* memory addresses together, which the GPU
  can service in a single wide memory transaction.
- **Synchronization.** `__syncthreads()` guarantees that all threads have finished loading
  the tile before any of them start computing, and that all have finished computing before
  the tile is overwritten on the next iteration.

The softmax and the `P·V` (output) stages are shared between the naive and optimized
versions so that the comparison isolates exactly the effect of optimizing the `Q·Kᵀ` step.

### 6.1 The softmax and output kernels

The two remaining stages run as their own kernels. The **softmax** kernel assigns one
thread to each row and, for that row, finds the maximum, exponentiates the (unmasked)
entries, accumulates their sum, and divides through — the same safe softmax as the CPU
version:

```cpp
int row = blockIdx.x;                      // one thread per row (threadIdx.x == 0)
float maxVal = -INFINITY;
for (int j = 0; j <= row; j++) maxVal = fmaxf(maxVal, scores[row*seq_len + j]);
float sum = 0.0f;
for (int j = 0; j <= row; j++) { float e = expf(scores[row*seq_len+j]-maxVal);
    scores[row*seq_len+j] = e; sum += e; }
for (int j = 0; j <= row; j++) scores[row*seq_len+j] /= sum;
```

This is deliberately simple and is **not** yet optimized — each row is processed serially by
a single thread. It is the clearest candidate for future work (Phase 5, parallel softmax),
and Part 2 shows that it is also where Triton wins the most.

The **output** kernel (`P·V`) uses a 2-D grid like the naive score kernel: the thread at
`(row, dim)` computes one element of the output by summing `P[row][j]·V[j][dim]` over the
allowed `j ≤ row`. It reads directly from global memory and is likewise a target for future
tiling (Phase 6).

### 6.2 Launch configuration

All score kernels launch with 16×16 thread blocks and a grid of
`⌈seq_len/16⌉ × ⌈seq_len/16⌉` blocks, so there is one thread per score and the grid exactly
covers the *n × n* matrix (with a boundary check for sequence lengths that are not multiples
of 16). The output kernel uses 16×16 blocks over a `⌈head_dim/16⌉ × ⌈seq_len/16⌉` grid, and
the softmax kernel launches `seq_len` one-thread blocks (one per row). Choosing 16×16 keeps
the shared-memory tiles small enough to allow several blocks to be resident per SM (good
occupancy) while still giving meaningful data reuse.

---

## 7. Phase 4 — Benchmarking methodology and results

**File:** `cuda_attention/benchmark.cu`

### 7.1 Methodology

The benchmark runs all three implementations on the same problem and times them:

- **Head dimension:** 64 (a realistic per-head size).
- **Sequence lengths:** 128, 256, 512, 1024.
- **Tiles:** 16×16.
- **CUDA timing:** CUDA **events** around each kernel, averaged over 20 iterations, so that
  only GPU execution time is measured.
- The CPU version is timed with a standard high-resolution clock.

### 7.2 Results (RTX 4060, from `benchmark.cu`)

| Sequence length | CPU (ms) | Naive CUDA (ms) | Optimized CUDA (ms) | Speed-up (CPU ÷ Opt) |
|---:|---:|---:|---:|---:|
| 128  | 0.3259  | 0.0556 | 0.0517 | 6.30× |
| 256  | 1.5480  | 0.0671 | 0.0481 | 32.16× |
| 512  | 7.0986  | 0.1849 | 0.1315 | 53.97× |
| 1024 | 30.7340 | 0.6447 | 0.4365 | 70.41× |

*(Re-running the benchmark on the same machine produced the same pattern with normal
run-to-run variation — for example a measured 67× at length 1024 in one run — which is
expected for short GPU kernels on a laptop whose clocks fluctuate.)*

### 7.3 Analysis

Two clear trends appear:

1. **The speed-up grows with sequence length.** At length 128 the GPU is only ~6× faster,
   but at 1024 it is ~70× faster. The reason is the *n²* nature of attention: longer
   sequences create far more independent work, which is exactly what fills the GPU's many
   cores and hides the fixed cost of launching a kernel and copying memory. At short
   lengths that fixed overhead dominates and the GPU is underutilized.

2. **The optimized kernel beats the naive kernel at every length**, and the gap is widest
   where it matters most (longer sequences), because that is where global-memory traffic
   is heaviest and shared-memory reuse saves the most.

The headline result is a **~70× speed-up** of the optimized CUDA attention over the CPU
baseline at a sequence length of 1024.

### 7.4 How the GPU timing is measured

Measuring GPU time correctly is subtle, because kernel launches are **asynchronous**: the
CPU queues a kernel and returns immediately, before the GPU has finished. Timing with a CPU
clock would therefore measure launch time, not execution time. The benchmark instead uses
**CUDA events** — markers inserted into the GPU's work queue. By recording an event before
and after the kernels and asking the GPU for the elapsed time between them, we measure the
true on-GPU duration. The input buffers and output buffers are allocated **once**, outside
the timed loop, so the measurement reflects the attention computation and not repeated
memory allocation. Host-to-device and device-to-host copies are likewise kept out of the
timed region, isolating the kernel cost.

### 7.5 Memory-bound vs compute-bound

The naive and optimized kernels do the *same* arithmetic; they differ only in how they use
memory. The fact that the optimized kernel is consistently faster tells us that the `Q·Kᵀ`
stage is **memory-bound** on this hardware — its speed is limited by how fast data reaches
the arithmetic units, not by the arithmetic itself. Tiling helps precisely because it
reduces and regularizes memory traffic. This is a recurring theme in GPU programming:
beyond a certain point, performance work is memory work. The per-stage analysis in Part 2
reinforces this — the biggest Triton gains come from reducing memory traffic (fusion, which
avoids writing the *n × n* matrix at all), not from faster arithmetic.

---

## 8. Phase 9 — GPT integration

**Folder:** `integration/`

Having built and benchmarked the kernels, we connect them back to the GPT so that the
model's attention can actually run on our own CUDA code.

### 8.1 Exposing the kernels to Python

The kernels are compiled as a **PyTorch C++/CUDA extension**
(`triton_attention/cuda_ext/`). The C++ binding exposes a single entry point:

```
attention(q, k, v, scores, out, optimized)
```

where `q, k, v` are 2-D contiguous FP32 CUDA tensors of shape `[T, head_dim]`, `scores` is
a preallocated `[T, T]` buffer and `out` is `[T, head_dim]`. The kernels used are copied
**verbatim** from `cuda_attention/benchmark.cu`, so the code running inside the GPT is
exactly the code we benchmarked.

### 8.2 A switchable attention backend

The integration defines an import-safe GPT (`integration/model.py`) whose every attention
head has a backend flag. A single call switches the whole model:

```python
model.set_attention_backend("cuda")    # or "pytorch"
```

- In the **PyTorch** backend the head computes attention the usual way (and this is what is
  used for training).
- In the **CUDA** backend the head calls our kernels, looping over `(batch, head)` because
  each kernel invocation handles one `[T, head_dim]` head.

Both backends scale attention by `1/√(head_size)` — the standard transformer scale, which
is exactly what the CUDA kernel computes — so the two produce the same result.

### 8.3 A deliberate design decision: training vs inference

Our CUDA kernels implement only the **forward** pass; there is no hand-written backward
(gradient) kernel. Back-propagation therefore requires the PyTorch backend. The honest and
correct integration is thus:

1. **Train** the GPT with the PyTorch backend (autograd computes gradients).
2. **Switch** the attention backend to CUDA.
3. **Verify** that the CUDA backend reproduces the PyTorch forward output.
4. **Generate** text with the CUDA backend.

### 8.4 Verification

`integration/run_integration.py` runs the full sequence end-to-end. After training it puts
the model in evaluation mode (dropout off) and compares the logits produced by the two
backends on the same input, reporting the maximum absolute difference. Because both
backends use identical maths, the difference is at the level of floating-point rounding
(well under `1e-3`), which confirms that the custom CUDA attention is correctly integrated.
The script also times one forward pass on each backend and generates a text sample, then
writes everything to `integration/results.json` for the UI to display.

---

## 9. The showcase UI and GPU properties

**Folder:** `ui/` (Streamlit)

A small web UI makes the results easy to demonstrate. It has four tabs:

1. **GPU** — the **live properties of the GPU** the kernels run on: name, compute
   capability, total memory, number of streaming multiprocessors, driver version,
   temperature and utilisation. This information is produced by `integration/gpu_info.py`,
   which reads the CUDA runtime through PyTorch when available and otherwise falls back to
   parsing `nvidia-smi`, so it works even without PyTorch installed. On the project machine
   it reports an **NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 8 GB, driver 610.47,** running
   with **CUDA 13.3**.
2. **CUDA Benchmark** — the CPU vs naive vs optimized table and bar charts.
3. **Triton Comparison** — the fair GPU comparison read from
   `triton_attention/results/benchmark_*.json`.
4. **GPT Integration** — the PyTorch-vs-CUDA error, the forward timings and the generated
   text sample.

The UI is launched with `streamlit run ui/app.py`. Requiring a live GPU-properties panel
was a specific goal of the project: it ties the abstract benchmark numbers to the concrete
piece of hardware that produced them.

---

## 10. Project structure and build

```
gpt/                PyTorch "GPT from scratch" (v2.py, bigram.py, input.txt, notebook)
cuda_attention/     C++/CUDA implementations + benchmark (the core project)
triton_attention/   Triton implementations, fair benchmark, results (beyond syllabus)
integration/        GPT + custom CUDA attention, GPU info, Streamlit data
ui/                 Streamlit showcase app
commands/           commands.txt — every command needed to run the project
docs/               this report and the presentation
```

Build requirements (confirmed on the project machine): an NVIDIA GPU with a current driver,
the **CUDA 13.3** toolkit (`nvcc`), **Visual Studio 2022** with the "Desktop development
with C++" workload (`cl.exe`), and **Python 3.13**. The C++/CUDA programs are compiled from
the "x64 Native Tools Command Prompt for VS 2022"; the Python parts use a virtual
environment described in `triton_attention/SETUP.md`. All commands are collected in
`commands/commands.txt`.

Build artifacts (`.exe`, `.obj`) are intentionally **not** committed — they are rebuilt
from source and are listed in `.gitignore`.

### 10.1 Design decisions, challenges and lessons learned

Several decisions shaped the project and are worth recording:

- **Keeping each phase as a separate program.** Rather than one configurable binary, the
  CPU, naive-CUDA and optimized-CUDA versions are separate files. This makes the *difference*
  between them easy to read and teach, which is the point of the exercise.
- **Optimizing one stage at a time.** Only `Q·Kᵀ` is tiled; softmax and `P·V` are shared
  between the naive and optimized versions. This isolates the effect of the optimization in
  the benchmark — any speed difference between naive and optimized is attributable to the
  tiled scores kernel alone, not to confounding changes elsewhere.
- **Reproducing, not replacing, the reference maths.** The integration scales attention by
  `1/√(head_size)` so that the CUDA backend reproduces the PyTorch backend exactly. A mismatch
  here would silently change the model's output; the automatic max-error check guards against
  it.
- **Accepting the forward-only limitation honestly.** Because there is no backward kernel,
  training stays on PyTorch and the CUDA kernels are used for inference. Presenting this
  clearly is more valuable than pretending the kernels can train.
- **A verbatim kernel copy with an automated check.** The PyTorch extension copies the
  kernels verbatim from `benchmark.cu`, and the Triton benchmark checks the text is identical.
  This removes any doubt that the benchmarked code and the integrated code are the same.

The main challenges were practical rather than algorithmic: getting the Windows toolchain to
agree on CUDA versions, the 260-character path limit (Part 2), and the general discipline of
measuring GPU time correctly with events and warm-up. The biggest conceptual lesson was that
**the interesting optimization is almost always about memory, not arithmetic** — tiling,
coalescing and fusion all win by moving less data, more regularly.

---

## 11. Limitations and future work

### 11.1 Current limitations

- **Only `Q·Kᵀ` is fully optimized.** The softmax and `P·V` stages are functional but not
  yet optimized in CUDA: the softmax kernel uses a single active thread per row, which is
  serial within a row.
- **Single head, FP32, forward-only.** The CUDA kernels process one head at a time, use
  32-bit floats, and have no backward pass, so training still relies on PyTorch.
- The GPT integration loops over `(batch, head)` in Python, which is fine for correctness
  but is not the fastest possible wiring.

### 11.2 Planned future work (Phases 5–10)

- **Phase 5 — Parallel softmax:** replace the one-thread-per-row softmax with a
  block-level parallel reduction (parallel max, parallel exp, parallel sum).
- **Phase 6 — Optimized `P·V`:** apply shared-memory tiling to the second matrix multiply.
- **Phase 7 — Multi-head attention:** process all heads together on the GPU.
- **Phase 8 — Kernel fusion:** combine the stages into one kernel to remove launch overhead
  and avoid writing the full *n × n* matrix to global memory (the FlashAttention idea,
  which Part 2 demonstrates in Triton).
- **Phase 9 — GPT integration:** now implemented for inference (this report).
- **Phase 10 — GPU profiling:** use Nsight tools to study occupancy and memory behaviour.

---
---

# PART 2 — CONTENT BEYOND SYLLABUS: TRITON

---

## 12. What Triton is and why we used it

**Triton** is a Python-embedded language and compiler from OpenAI for writing GPU kernels.
A kernel is an ordinary-looking Python function decorated with `@triton.jit`; Triton
compiles it *just-in-time* to **PTX** — the same GPU assembly that `nvcc` produces — and
then to machine code for whichever GPU it runs on. Triton is not a toy: PyTorch's own
`torch.compile` generates Triton kernels under the hood.

We included Triton as a beyond-syllabus extension to answer a practical question that the
CUDA work naturally raises: *now that we have hand-written and hand-optimized these kernels
in CUDA, how do they compare — in both speed and development effort — against a modern
kernel language and against PyTorch's own highly-tuned attention?* All Triton code lives in
`triton_attention/`, and the original CUDA files were not modified.

## 13. Block-level vs thread-level programming

The fundamental difference between CUDA and Triton is the unit the programmer controls:

| | CUDA (our kernels) | Triton |
|---|---|---|
| Unit you program | one **thread** (`threadIdx`, `blockIdx`) | one **program** handling a whole **tile** of data (`tl.program_id`) |
| Data handled | scalars, e.g. `Q[row*d + k]` | tensors, e.g. a whole `[16,16]` tile via `tl.load` |
| Shared memory | manual: `__shared__`, `__syncthreads()` | automatic: the compiler decides |
| Matrix multiply | hand-written loop over the tile | `tl.dot(a, b)` (can use tensor cores) |
| Bounds checks | `if (row < n && col < n)` per thread | a `mask=` argument on load/store |
| Coalescing, pipelining | by hand | done by the compiler |
| Tuning tile sizes | edit constants, recompile | `@triton.autotune` tries configs on the real GPU |

In CUDA we tell each individual thread what to do; in Triton we describe what to do with an
entire block of data and let the compiler assign the threads. The same tiled `Q·Kᵀ` step is
about ten lines of index arithmetic and shared-memory management in CUDA, and a handful of
`tl.load` / `tl.dot` calls in Triton.

## 14. Implementation A — Triton tiled (3-kernel port)

**File:** `triton_attention/01_triton_tiled_qkt.py`

This is an *algorithm-for-algorithm* port of our optimized CUDA version. It keeps the same
structure — 16×16 tiles for `Q·Kᵀ`, the full *n × n* score matrix materialized in global
memory, and then separate softmax and `P·V` kernels — but expresses each stage in Triton.
It exists specifically so that the comparison against CUDA is fair: the *same* algorithm,
written in the two languages.

## 15. Implementation B — Triton fused (online softmax)

**File:** `triton_attention/02_triton_fused_attention.py`

This is the modern, high-performance approach in the **FlashAttention** style, expressed as
a **single fused kernel**. Each program loads one tile of Q and then loops over the tiles of
K and V, maintaining a running maximum and running sum so that the softmax is computed
*incrementally* (an **online softmax**) without ever storing the full *n × n* score matrix.
It also skips the K/V tiles that lie entirely above the causal diagonal, since those
positions are masked out anyway. Tile sizes are chosen automatically by `@triton.autotune`.

The two advantages over the staged approach are decisive for large sequences:

- **No *n × n* matrix is ever written to global memory**, which removes the dominant memory
  cost and the dominant memory footprint.
- **Roughly half the work is skipped** because of the causal structure.

### 15.1 The online softmax, in detail

The staged approach needs the whole score row before it can compute the softmax (it must
know the row maximum and the row sum). The fused kernel cannot afford to keep the whole row,
so it computes the softmax **incrementally** as it streams over the K/V tiles. For each new
tile it:

1. finds the maximum `m_new` of the running maximum `m` and the new tile's scores;
2. rescales the running sum and the running output by `exp(m − m_new)` to account for the
   new, larger maximum;
3. adds the new tile's contribution to the sum and to the weighted output;
4. updates `m ← m_new`.

In pseudocode, for output accumulator `O`, running sum `l` and running max `m`:

```
for each K/V tile:
    s      = (Q · Kᵀ_tile) / √d          # scores for this tile (with causal mask)
    m_new  = max(m, rowmax(s))
    p      = exp(s − m_new)              # safe exponent against the new max
    alpha  = exp(m − m_new)             # correction factor for what we already have
    l      = l * alpha + rowsum(p)
    O      = O * alpha + p · V_tile
    m      = m_new
O = O / l                                # final normalization
```

Because `O` and `l` are rescaled every time the maximum grows, the final result is exactly
the same as a full softmax — but the *n × n* matrix is never materialized, and the kernel
touches each K/V tile only once. This is the essence of **FlashAttention**, and expressing
it in Triton takes only 36 lines of kernel code.

### 15.2 CUDA graphs and autotuning

Two Triton/PyTorch mechanisms are important to the benchmark:

- **Autotuning.** `@triton.autotune` compiles the kernel with several candidate tile sizes
  and `num_warps`/`num_stages` settings, runs each on the actual GPU, and keeps the fastest.
  This happens on the first call and is excluded from timing (it runs during warm-up).
- **CUDA graphs.** A CUDA graph records a sequence of kernel launches once and then
  *replays* it with almost no CPU involvement. The benchmark times both **eager** mode
  (normal Python calls, which include the CPU cost of launching each kernel) and **graph**
  mode (pure GPU time). The two modes matter because at small sequence lengths the Python
  launch cost can be larger than the GPU work itself, so eager and graph numbers tell
  different, both-valid, stories.

## 16. Numerical precision: TF32, IEEE FP32, FP16

Our CUDA code is strictly 32-bit float (FP32), so to compare fairly the Triton FP32 numbers
must also be true FP32. On Ampere/Ada GPUs, Triton's `tl.dot` uses **TF32** by default — a
reduced-precision mode that is faster but less accurate — so we force full precision with
`input_precision="ieee"`. The smoke test makes the difference concrete: a 64×64 FP32 product
has an error of `2.7e-6` with `ieee` versus `7.3e-3` with TF32. A separate **FP16** variant
of the fused kernel is reported *on its own*, because comparing FP16 against FP32 CUDA would
not be a like-for-like comparison.

## 17. Fair benchmarking methodology

**File:** `triton_attention/benchmark_triton.py`

Every GPU version is timed from the same Python process in exactly the same way:

- Identical Q, K, V tensors (independent uniform `[-1, 1]`, fixed seed), `d = 64`, already
  on the GPU.
- Output and score buffers allocated **once**, outside the timed region (as in
  `benchmark.cu`).
- **Every version is checked against PyTorch's `scaled_dot_product_attention` before it is
  timed** — a version that gives wrong answers is never benchmarked.
- CUDA events around 100 back-to-back calls, after 10 warm-up calls, repeated for 5 rounds,
  with the **median round** reported (all rounds saved to JSON).
- Two timing modes: **CUDA-graph replay** (pure GPU time) and **eager** (normal Python
  calls, which include CPU launch overhead).

A notable fairness detail: the benchmark reads the kernel text out of
`cuda_attention/benchmark.cu` and checks that the extension's kernels are **character-for-
character identical**, so there is no chance of accidentally benchmarking different code.

## 18. Correctness results

Maximum absolute error against PyTorch SDPA (FP32, causal, `d = 64`):

| Seq len | naive CUDA | optimized CUDA | Triton tiled | Triton fused |
|---:|---:|---:|---:|---:|
| 128  | 2.24e-07 | 2.24e-07 | 2.98e-07 | 2.38e-07 |
| 256  | 2.24e-07 | 2.24e-07 | 2.09e-07 | 2.09e-07 |
| 512  | 1.53e-07 | 1.53e-07 | 2.09e-07 | 1.79e-07 |
| 1024 | 1.94e-07 | 1.94e-07 | 1.79e-07 | 1.79e-07 |

All FP32 versions agree with PyTorch to rounding level (< `3e-7`), confirming that every
implementation computes the same mathematical result.

## 19. Performance results

> **Note on hardware.** The fair Triton benchmark numbers below were measured on the
> **development laptop (NVIDIA RTX 3050 6 GB, sm_86)**, which is where the Triton toolchain
> was set up. They are clearly labelled as such. The equivalent table for the project's
> **RTX 4060** has not yet been generated; it is produced by running
> `triton_attention/benchmark_triton.py` on that machine. The RTX 4060 numbers in Part 1
> come from `benchmark.cu`, which uses a different timer (20 iterations, no CUDA graphs) and
> therefore should not be mixed with the table below.

**CUDA-graph replay (pure GPU time), RTX 3050, FP32, `d = 64`, ms per call:**

| Seq len | naive CUDA | optimized CUDA | Triton tiled | Triton fused | PyTorch SDPA |
|---:|---:|---:|---:|---:|---:|
| 128  | 0.0340 | 0.0239 | 0.0105 | 0.0156 | 0.0263 |
| 256  | 0.0780 | 0.0545 | 0.0238 | 0.0276 | 0.0471 |
| 512  | 0.2506 | 0.1767 | 0.0718 | 0.0535 | 0.0895 |
| 1024 | 0.9193 | 0.6422 | 0.3142 | 0.1422 | 0.1768 |

**Speed-up over our optimized CUDA (graph mode; > 1 means faster):**

| Seq len | Triton tiled | Triton fused | PyTorch SDPA |
|---:|---:|---:|---:|
| 128  | 2.26× | 1.53× | 0.91× |
| 256  | 2.29× | 1.97× | 1.16× |
| 512  | 2.46× | 3.31× | 1.97× |
| 1024 | 2.04× | 4.52× | 3.63× |

The fused Triton kernel is faster than PyTorch's own FP32 attention at every tested length
in graph mode — for example 0.1422 ms vs 0.1768 ms at length 1024.

## 20. Per-stage and lines-of-code analysis

**Per-stage time at length 1024 (graph mode, ms, RTX 3050):**

| Stage | naive CUDA | optimized CUDA | Triton tiled |
|---|---:|---:|---:|
| 1. Q·Kᵀ + scale + mask | 0.5048 | 0.2201 | 0.2017 |
| 2. softmax | 0.2852 | 0.2852 | 0.0546 |
| 3. P·V | 0.1422 | 0.1422 | 0.0597 |

This table is the most instructive result in Part 2. For the **one stage we optimized the
same way in both languages** (tiled `Q·Kᵀ`), Triton and our shared-memory CUDA kernel are
within about 9% (0.2017 vs 0.2201 ms). Most of Triton tiled's overall advantage comes from
the two stages we have **not** yet optimized in CUDA: the Triton softmax is ~5× faster than
our one-thread-per-row CUDA softmax, and the Triton `P·V` is ~2.4× faster. These are exactly
the planned CUDA Phases 5 and 6 — in Triton they came almost for free from the block model.

**Lines of device (kernel) code, excluding blanks and comments:**

| Implementation | Kernel lines | Kernels | Manual shared memory |
|---|---:|---:|:---:|
| naive CUDA | 69 | 3 | no |
| optimized CUDA | 91 | 3 | yes |
| Triton tiled | 50 | 3 | no (compiler) |
| Triton fused | 36 | 1 | no (compiler) |

## 21. Discussion: performance vs development effort

1. **Same algorithm, same speed.** For the identically-written tiled `Q·Kᵀ`, Triton matches
   hand-tuned CUDA within ~9%, but with no `__shared__` arrays, no `__syncthreads()` and no
   per-thread index arithmetic.
2. **The block model makes the next optimizations cheap.** The optimizations we still have
   to hand-write in CUDA (parallel softmax, tiled `P·V`) are essentially automatic in
   Triton, which is where most of its lead comes from.
3. **Fusion is the biggest win and is short in Triton.** The 36-line fused kernel is ~4.5×
   faster than our optimized CUDA at length 1024 and beats PyTorch's FP32 attention, because
   it never writes the *n × n* matrix and skips the masked half of the work. Doing the same
   in CUDA (the planned Phase 8) would mean hand-writing online softmax, tiling for both
   matmuls, and the tile-skipping logic.
4. **CUDA gives full control; Triton trades some control for productivity.** CUDA exposes
   every thread and every byte of memory, which is ideal for learning *how* the hardware
   works — the explicit goal of this course. Triton hides those details and lets the
   compiler and the autotuner make the decisions, which is ideal for getting a fast kernel
   quickly.

**Summary:** for the same algorithm, Triton matches hand-written CUDA with far less code;
its larger benefit is that the next optimizations are short to write. On the development
GPU, moving from our optimized CUDA to the Triton fused kernel took attention from 0.6422 ms
to 0.1422 ms at length 1024 (FP32).

## 22. Windows setup challenges

Setting up Triton on native Windows surfaced real, documented problems (full details in
`triton_attention/SETUP.md`):

- Triton has no official Windows release; we used the community **`triton-windows`** build,
  which bundles `ptxas` and a small C compiler so kernels JIT-compile without a full toolkit.
- PyTorch and `nvcc` must share the same CUDA **major** version (here 13.x) or the extension
  build is rejected — which is why a separate `cu130` virtual environment is used rather than
  the repository's default `cu121` PyTorch.
- The Windows **260-character path limit** broke both the Triton import and the extension
  build (headers "not found" although present); a short virtual-environment path fixed both.
- `tl.dot`'s default TF32 behaviour silently lowers FP32 accuracy unless
  `input_precision="ieee"` is set.

---
---

# Conclusion

This project implemented Transformer self-attention from first principles and accelerated
it on the GPU in clear, measurable stages. The sequential CPU baseline was parallelized
with CUDA (one thread per score) and then optimized with shared-memory tiling, reaching a
**~70× speed-up over the CPU at a sequence length of 1024** on the project's RTX 4060. The
custom kernels were then integrated back into a GPT model through a PyTorch extension, with
a switchable attention backend and an automatic check confirming that the CUDA attention
reproduces the reference output; a Streamlit UI showcases the results and the live GPU
properties.

Beyond the syllabus, the same attention was re-implemented in OpenAI Triton — including a
FlashAttention-style fused kernel — and compared fairly against hand-written CUDA and
PyTorch's own attention. The comparison showed that for an identical algorithm Triton
matches hand-tuned CUDA, and that its block-level model makes the remaining optimizations
(parallel softmax, tiled `P·V`, fusion) short to express — the fused kernel reaching ~4.5×
over our optimized CUDA on the development GPU. Together, the two parts give both a
ground-up understanding of *why* GPUs accelerate attention and a practical view of the
trade-off between hand-written CUDA and a modern kernel compiler.

# References

1. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
2. A. Karpathy, "Let's build GPT: from scratch, in code, spelled out," 2023.
3. T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with
   IO-Awareness," NeurIPS, 2022.
4. OpenAI, "Triton: an open-source language and compiler for GPU programming."
5. NVIDIA, "CUDA C++ Programming Guide," CUDA Toolkit 13.3.
6. PyTorch, "torch.nn.functional.scaled_dot_product_attention" documentation.

# Appendix A — How to run everything

See `commands/commands.txt` for the full, copy-pasteable command list (prerequisite checks,
the four CUDA programs, the Triton benchmark, the GPT integration, and the Streamlit UI).

# Appendix B — File map

| Path | Contents |
|---|---|
| `gpt/` | The PyTorch "GPT from scratch" model and its training text. |
| `cuda_attention/` | CPU, naive-CUDA and optimized-CUDA attention + the benchmark. |
| `triton_attention/` | Triton implementations, the fair benchmark and the results. |
| `integration/` | The GPT + custom-CUDA-attention integration and GPU-info utility. |
| `ui/` | The Streamlit showcase application. |
| `commands/` | `commands.txt`. |
| `docs/` | This report and the presentation. |

# Appendix C — Glossary

| Term | Meaning |
|---|---|
| **Attention score** | An entry of the `Q·Kᵀ` matrix: how strongly one token attends to another. |
| **Causal mask** | A rule that stops a token from attending to later tokens (sets those scores to −∞). |
| **CUDA** | NVIDIA's platform/language for general-purpose GPU programming in C++. |
| **CUDA event** | A timestamp marker in the GPU's work queue, used to measure true GPU time. |
| **CUDA graph** | A recorded sequence of GPU launches that can be replayed with almost no CPU overhead. |
| **Coalescing** | Combining the memory reads of a warp into one wide transaction when addresses are consecutive. |
| **FlashAttention** | A fused attention algorithm that uses an online softmax and never stores the *n × n* matrix. |
| **Global memory** | The GPU's large but relatively slow main memory. |
| **Grid / block / thread** | The three levels of the CUDA execution hierarchy. |
| **Head dimension (dₖ)** | The size of each query/key/value vector within one attention head. |
| **Kernel** | A function (`__global__`) that runs on the GPU. |
| **Occupancy** | The fraction of the maximum resident warps that are active on an SM. |
| **Online softmax** | A softmax computed incrementally with a running max and sum, without the full row. |
| **PTX** | NVIDIA's intermediate GPU assembly; both `nvcc` and Triton compile to it. |
| **SDPA** | `scaled_dot_product_attention`, PyTorch's built-in optimized attention. |
| **Shared memory** | Small, fast, per-block memory that threads in a block share; a programmer-managed cache. |
| **SM (Streaming Multiprocessor)** | A core group on the GPU that runs many warps concurrently. |
| **Softmax** | Turns a row of scores into a probability distribution that sums to 1. |
| **TF32** | A reduced-precision mode for matrix multiply on Ampere/Ada GPUs; faster but less accurate than FP32. |
| **Tiling** | Loading a small block (tile) of data into shared memory so many threads can reuse it. |
| **Triton** | A Python-embedded language/compiler for GPU kernels, where you program a tile of data at a time. |
| **Warp** | A group of 32 threads that execute the same instruction together (SIMT). |

