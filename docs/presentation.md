---
marp: true
title: Optimizing Transformer Self-Attention with CUDA
paginate: true
---

<!--
PRESENTATION DECK (Markdown)
Each "---" starts a new slide. This is Marp-compatible: you can render it to slides
with the "Marp for VS Code" extension or the Marp CLI, or just present it as Markdown.
Lines in italics labelled "Speaker notes" are talking points, not slide text.
Suggested length: ~18 slides, ~12-15 minutes.
-->

# Optimizing Transformer Self-Attention using CUDA

### From CPU → Naive CUDA → Optimized CUDA → Triton → GPT integration

GPU Computing · Semester 7
Target GPU: NVIDIA RTX 4060 Laptop (sm_89) · CUDA 13.3

*Speaker notes: Introduce the one-line goal — take the most expensive part of a GPT, build it from scratch, and make it fast on the GPU, measuring every step.*

---

## The problem

- Transformers/GPT spend most of their compute in **self-attention**.
- Attention compares **every token with every other token** → an *n × n* matrix.
- Cost grows **quadratically**: `O(n² · d)`. Double the sequence → 4× the work.
- But those *n²* comparisons are **independent** → perfect for a GPU.

*Speaker notes: This slide motivates everything. The quadratic cost is both the pain and the opportunity.*

---

## What is self-attention?

Each token has three vectors: **Query**, **Key**, **Value**.

$$\text{Attention}(Q,K,V) = \text{Softmax}\!\left(\tfrac{QK^{T}}{\sqrt{d_k}} + M\right)V$$

Four steps:
1. **Q·Kᵀ / √d** — similarity of every query to every key
2. **Causal mask** — a token sees only itself + earlier tokens
3. **Softmax** — turn each row into probabilities
4. **· V** — weighted average of value vectors

*Speaker notes: Keep this concrete. The causal mask is what makes it a GPT (no peeking at the future).*

---

## Where it lives in a GPT

```python
wei = q @ k.transpose(-2,-1) * head_size**-0.5  # 1. Q·Kᵀ/√d
wei = wei.masked_fill(tril==0, float('-inf'))    # 2. causal mask
wei = F.softmax(wei, dim=-1)                      # 3. softmax
out = wei @ v                                     # 4. ·V
```

- The repo includes a small "GPT from scratch" (`gpt/v2.py`) for context.
- Our project **reimplements these four lines in CUDA** and makes them fast.

*Speaker notes: Tie the maths to real model code — these four PyTorch lines are exactly what we rebuild.*

---

## GPU in one slide

- **Thread** → one unit of work (one score).
- **Block** → group of threads sharing fast **shared memory** + `__syncthreads()`.
- **Grid** → all blocks covering the problem.

Memory hierarchy (speed):
**registers > shared memory >> global memory (slow!)**

> Key optimization: move reused data from **global** → **shared** memory.

*Speaker notes: This is the mental model the whole optimization rests on.*

---

## Phase 1 — CPU baseline

**File:** `cuda_attention/01_cpu_attention.cpp`

- Plain C++ nested loops; tiny 4×3 example you can check by hand.
- Four functions = the four steps.
- Fully **sequential** — one score at a time on one core.
- Purpose: a correct reference to compare against.

*Speaker notes: Show the lower-triangular attention weights from the causal mask if time allows.*

---

## Phase 2 — Naive CUDA

**File:** `cuda_attention/02_naive_cuda_attention.cu`

- **One thread per attention score**: thread `(row,col)` computes `S[row][col]`.
- *n²* scores now computed in parallel.
- Full pipeline on GPU: copy H→D, scores kernel, softmax, output, copy D→H.
- **Weakness:** every thread re-reads Q/K from **slow global memory**.

*Speaker notes: Emphasize the redundant global-memory reads — that's what the next phase fixes.*

---

## Phase 3 — Optimized CUDA (shared memory + tiling)

**File:** `cuda_attention/03_optimized_attention.cu`

- A block loads a **16×16 tile** of Q and K into **shared memory**, once.
- All threads in the block reuse the tile → ~16× fewer global reads.
- `__syncthreads()` coordinates load vs compute.
- Only `Q·Kᵀ` is optimized (softmax & ·V shared with naive, to isolate the effect).

*Speaker notes: This is the core CUDA technique of the course — tiling for data reuse.*

---

## Phase 4 — Benchmark results (RTX 4060)

d = 64, from `cuda_attention/benchmark.cu`

| Seq len | CPU (ms) | Naive (ms) | Optimized (ms) | Speed-up |
|---:|---:|---:|---:|---:|
| 128  | 0.33  | 0.056 | 0.052 | 6.3× |
| 256  | 1.55  | 0.067 | 0.048 | 32× |
| 512  | 7.10  | 0.185 | 0.132 | 54× |
| 1024 | 30.73 | 0.645 | 0.437 | **70×** |

*Speaker notes: The headline number is ~70× at length 1024.*

---

## Why the speed-up grows with length

```
Seq length     Speed-up
  128            6.3×
  256           32×
  512           54×
 1024           70×
```

- Attention is `O(n²·d)` → longer sequences = far more parallel work.
- More work fills the GPU's thousands of cores and hides launch/copy overhead.
- At short lengths, fixed overhead dominates → smaller win.

*Speaker notes: This is the single most important analytical point of Part 1.*

---

## Phase 9 — GPT integration

**Folder:** `integration/`

- Kernels exposed to Python as a **PyTorch extension** (verbatim from `benchmark.cu`).
- GPT head has a **switchable backend**: `set_attention_backend('pytorch' | 'cuda')`.
- Kernels are **forward-only** → train with PyTorch, run inference with CUDA.
- **Verified:** CUDA backend reproduces PyTorch output (max error < 1e-3).

*Speaker notes: Be honest about the forward-only design — it's the correct engineering choice, not a shortcut.*

---

## Showcase UI + GPU properties

**Folder:** `ui/` (Streamlit) — `streamlit run ui/app.py`

- **GPU tab:** live properties — **RTX 4060 Laptop, sm_89, 8 GB, driver 610.47, CUDA 13.3**.
- **CUDA Benchmark tab:** the CPU-vs-CUDA table + charts.
- **Triton tab:** fair GPU comparison from results JSON.
- **GPT Integration tab:** PyTorch-vs-CUDA error + generated text.

*Speaker notes: Demo live if a projector is available — the GPU tab is a nice "here is the real hardware" moment.*

---

## Beyond syllabus — Triton

- **Triton** = Python language + JIT compiler for GPU kernels (used inside PyTorch).
- You program a **whole tile** (`tl.load`, `tl.dot`), not individual threads.
- Compiler handles shared memory, coalescing, pipelining; `@triton.autotune` picks tile sizes.
- Question: how does hand-written CUDA compare to Triton and to PyTorch's own attention?

*Speaker notes: Frame Triton as the "modern" counterpoint to the hand-written CUDA.*

---

## Two Triton implementations

- **Triton tiled** (`01_...py`) — same algorithm as our optimized CUDA (3 kernels, full *n×n* matrix). For a fair comparison.
- **Triton fused** (`02_...py`) — **FlashAttention style**, one kernel:
  - **online softmax** (running max & sum)
  - **never stores the *n×n* matrix**
  - **skips the masked half** of the work
  - autotuned tile sizes

*Speaker notes: Fusion is the big idea — it's what makes modern attention fast.*

---

## Triton vs CUDA — results (dev GPU, RTX 3050)

Graph mode, FP32, d=64, ms/call. *(RTX 4060 Triton table: run `benchmark_triton.py` to fill in.)*

| Seq len | optimized CUDA | Triton tiled | Triton fused | PyTorch SDPA |
|---:|---:|---:|---:|---:|
| 512  | 0.177 | 0.072 | 0.054 | 0.090 |
| 1024 | 0.642 | 0.314 | **0.142** | 0.177 |

- Same algorithm (tiled Q·Kᵀ): Triton ≈ CUDA (within ~9%).
- Fused Triton: **4.5× over our optimized CUDA** at 1024, beats PyTorch FP32.

*Speaker notes: Be explicit these are RTX 3050 dev numbers; the 4060 table is pending.*

---

## Performance vs development effort

| Implementation | Kernel lines | Manual shared mem |
|---|---:|:---:|
| optimized CUDA | 91 | yes |
| Triton tiled | 50 | no (compiler) |
| Triton fused | 36 | no (compiler) |

- Same algorithm → same speed, **far less code** in Triton.
- CUDA = full control (great for *learning the hardware*).
- Triton = compiler does the hard parts (great for *getting fast quickly*).

*Speaker notes: This is the thoughtful takeaway — not "X beats Y" but a trade-off.*

---

## Conclusion

- Built attention from scratch and accelerated it: **CPU → CUDA → optimized CUDA**.
- **~70× speed-up** over CPU at length 1024 on the RTX 4060.
- Integrated the kernels into a GPT (verified correct) + a Streamlit showcase.
- Beyond syllabus: Triton matches hand-tuned CUDA with less code; fusion is the big win.
- **Future:** parallel softmax, tiled ·V, multi-head, full kernel fusion in CUDA, profiling.

*Speaker notes: End on the future-work roadmap — it shows the project is a foundation, not a dead end.*

---

# Thank you — Questions?

Repo: `Optimization-of-transformers-using-CUDA` · branch `feature/triton-attention`
Report: `docs/project_report.md` · Commands: `commands/commands.txt`

*Speaker notes: Offer to run the live demo (`streamlit run ui/app.py`) or the benchmark.*
