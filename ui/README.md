# Showcase UI (Streamlit)

A small web UI that showcases the project's results and the GPU it runs on.

## Tabs

1. **GPU** — live properties of the GPU (name, compute capability, memory, SM count,
   driver, temperature, utilisation), read via `integration/gpu_info.py`.
2. **CUDA Benchmark** — CPU vs naive CUDA vs optimized CUDA (`cuda_attention/benchmark.cu`).
3. **Triton Comparison** — the fair GPU comparison from
   `triton_attention/results/benchmark_*.json` (CUDA vs Triton vs PyTorch SDPA).
4. **GPT Integration** — PyTorch-vs-CUDA attention error, forward timings and a text
   sample from `integration/results.json`.

## Install + run

```bat
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

The GPU tab works immediately. The Triton and GPT Integration tabs fill in once you
have run `triton_attention/benchmark_triton.py` and `integration/run_integration.py`
respectively (the UI shows the exact command when a result file is missing).
