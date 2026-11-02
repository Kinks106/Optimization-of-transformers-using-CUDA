# GPT Integration (Phase 9)

This folder connects the two halves of the project: the **GPT from scratch**
transformer and our **custom CUDA attention kernels**. The GPT's attention head can
run on PyTorch (for training) or on our CUDA kernels (for inference / generation),
and we verify that both produce the same output.

## Why training stays on PyTorch

Our CUDA kernels (in `../triton_attention/cuda_ext/`, copied verbatim from
`../cuda_attention/benchmark.cu`) are **forward-only** — there is no backward kernel.
Back-propagation therefore uses the PyTorch backend. After training, we switch the
attention backend to CUDA for the forward pass and for generation.

Both backends scale attention by `1/sqrt(head_size)` (the standard transformer
scale, which is exactly what the CUDA kernel computes), so their outputs match to
floating-point rounding.

## Files

| File | Purpose |
|---|---|
| `gpu_info.py` | Reports the GPU this project runs on (works with or without PyTorch). |
| `cuda_attention_backend.py` | Builds the CUDA extension and exposes `cuda_causal_attention(q, k, v)`. |
| `model.py` | Import-safe GPT with `set_attention_backend('pytorch' \| 'cuda')`. |
| `run_integration.py` | Train → verify CUDA matches PyTorch → generate with CUDA → save `results.json`. |

## Run

From the repo root, in a shell where the CUDA extension can build (see
`../triton_attention/SETUP.md` — needs the VS "x64 Native Tools" prompt + a CUDA
toolkit whose major version matches PyTorch):

```bat
.venv-triton\Scripts\python.exe integration\run_integration.py
```

Quick options:

```bat
:: PyTorch backend only (no extension build needed)
.venv-triton\Scripts\python.exe integration\run_integration.py --skip-cuda

:: Larger model, more training
.venv-triton\Scripts\python.exe integration\run_integration.py --max-iters 2000 --n-embd 384 --n-head 6 --n-layer 6

:: Just print the GPU properties
.venv-triton\Scripts\python.exe integration\gpu_info.py
```

Outputs: `results.json` (GPU info, PyTorch-vs-CUDA max error, forward timings, a
text sample) which the Streamlit UI reads, and `gpt_weights.pt` (trained weights).
