"""app.py - Streamlit UI to showcase the project's results.

Four tabs:
  1. GPU            - live properties of the GPU running the kernels (gpu_info.py)
  2. CUDA Benchmark - CPU vs naive vs optimized CUDA (from cuda_attention/benchmark.cu)
  3. Triton Compare - fair GPU comparison from triton_attention/results/*.json
  4. GPT Integration- PyTorch-vs-CUDA attention check + generated text (integration/results.json)

Run from the repo root:
    streamlit run ui/app.py
"""

import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_INTEGRATION = os.path.join(_REPO, "integration")
_TRITON_RESULTS = os.path.join(_REPO, "triton_attention", "results")
if _INTEGRATION not in sys.path:
    sys.path.insert(0, _INTEGRATION)

# gpu_info has no hard dependency on torch, so this import is safe either way.
try:
    from gpu_info import get_gpu_info
except Exception as exc:                      # pragma: no cover
    get_gpu_info = None
    _GPU_IMPORT_ERROR = str(exc)

# Static CUDA-only benchmark from cuda_attention/benchmark.cu (RTX 4060, d=64, 20 iters).
# Always available so the UI shows something even before anything is run.
CUDA_BENCHMARK_CU = pd.DataFrame({
    "Seq length": [128, 256, 512, 1024],
    "CPU (ms)": [0.3259, 1.5480, 7.0986, 30.7340],
    "Naive CUDA (ms)": [0.0556, 0.0671, 0.1849, 0.6447],
    "Optimized CUDA (ms)": [0.0517, 0.0481, 0.1315, 0.4365],
    "Speedup (CPU/Opt)": [6.30, 32.16, 53.97, 70.41],
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_triton_result_files():
    if not os.path.isdir(_TRITON_RESULTS):
        return []
    return sorted(f for f in os.listdir(_TRITON_RESULTS)
                  if f.startswith("benchmark_") and f.endswith(".json"))


@st.cache_data(show_spinner=False)
def load_json(path, _mtime):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def triton_timing_table(data, mode, group="fp32"):
    """Pivot the results list into seq_len x version (median ms) for one mode/group."""
    rows = [r for r in data.get("results", []) if r["mode"] == mode and r["group"] == group]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="seq_len", columns="version", values="median_ms")
    # Keep a sensible column order if the expected versions are present.
    order = ["naive CUDA", "optimized CUDA", "Triton tiled", "Triton fused", "PyTorch SDPA"]
    cols = [c for c in order if c in pivot.columns] + [c for c in pivot.columns if c not in order]
    return pivot[cols]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CUDA Attention — Results", page_icon="⚡", layout="wide")
st.title("⚡ Optimizing Transformer Self-Attention with CUDA")
st.caption("CPU → naive CUDA → shared-memory CUDA → Triton → GPT integration. "
           "Showcase of results and the GPU they run on.")

tab_gpu, tab_cuda, tab_triton, tab_gpt = st.tabs(
    ["🖥️ GPU", "📊 CUDA Benchmark", "🔬 Triton Comparison", "🤖 GPT Integration"]
)

# --- Tab 1: GPU ------------------------------------------------------------
with tab_gpu:
    st.subheader("GPU this project runs on")
    if get_gpu_info is None:
        st.error(f"Could not import gpu_info: {_GPU_IMPORT_ERROR}")
    else:
        if st.button("🔄 Refresh"):
            st.rerun()
        info = get_gpu_info()
        if not info["available"]:
            st.warning("No CUDA-capable GPU detected (or no driver / nvidia-smi).")
        else:
            st.caption(f"Source: {info['source']}"
                       + (f" · PyTorch {info['torch_version']} (CUDA {info['cuda_version']})"
                          if info.get("torch_version") else "")
                       + (f" · nvcc {info['nvcc_version']}" if info.get("nvcc_version") else ""))
            for g in info["gpus"]:
                st.markdown(f"**GPU {g.get('index', 0)}: {g.get('name')}**")
                c = st.columns(4)
                c[0].metric("Compute capability", g.get("compute_capability", "—"))
                c[1].metric("Total memory", f"{g.get('total_memory_gb', '—')} GB")
                c[2].metric("Streaming multiprocessors", g.get("multi_processor_count", "—"))
                c[3].metric("Driver", g.get("driver_version", "—"))
                c2 = st.columns(4)
                c2[0].metric("Temperature", f"{g.get('temperature_c', '—')} °C")
                c2[1].metric("Utilisation", f"{g.get('utilization_pct', '—')} %")
                if g.get("memory_used_mb"):
                    c2[2].metric("Memory used", f"{g.get('memory_used_mb')} MB")
                if g.get("power_draw_w"):
                    c2[3].metric("Power draw", f"{g.get('power_draw_w')} W")
            with st.expander("Raw info (JSON)"):
                st.json(info)

# --- Tab 2: CUDA benchmark -------------------------------------------------
with tab_cuda:
    st.subheader("CPU vs Naive CUDA vs Optimized CUDA")
    st.caption("From `cuda_attention/benchmark.cu` (head dim 64). "
               "Lower time is better; speedup is CPU ÷ optimized CUDA.")
    st.dataframe(CUDA_BENCHMARK_CU, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Time per call (ms) — log scale shows the gap**")
        chart_df = CUDA_BENCHMARK_CU.set_index("Seq length")[
            ["CPU (ms)", "Naive CUDA (ms)", "Optimized CUDA (ms)"]]
        st.bar_chart(chart_df)
    with c2:
        st.markdown("**Speedup over CPU (optimized CUDA)**")
        st.bar_chart(CUDA_BENCHMARK_CU.set_index("Seq length")[["Speedup (CPU/Opt)"]])
    st.info("Speedup grows with sequence length because self-attention is O(n²·d): "
            "more parallel work for the GPU to hide launch cost.")

# --- Tab 3: Triton comparison ---------------------------------------------
with tab_triton:
    st.subheader("Fair GPU comparison (CUDA vs Triton vs PyTorch SDPA)")
    files = list_triton_result_files()
    if not files:
        st.warning("No Triton benchmark results found yet in `triton_attention/results/`.")
        st.code("python triton_attention/benchmark_triton.py", language="bat")
    else:
        choice = st.selectbox("Results file", files,
                              help="Generated by triton_attention/benchmark_triton.py")
        path = os.path.join(_TRITON_RESULTS, choice)
        data = load_json(path, os.path.getmtime(path))
        st.caption(f"GPU: **{data.get('gpu')}** · torch {data.get('torch')} · "
                   f"triton {data.get('triton')}")
        modes = sorted({r["mode"] for r in data.get("results", [])})
        mode = st.radio("Timing mode", modes, horizontal=True,
                        help="graph = GPU time only (CUDA graph replay); "
                             "eager = normal Python calls (includes launch overhead)")
        table = triton_timing_table(data, mode)
        if table is not None:
            st.markdown("**Time per attention call (ms)** — lower is better")
            st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
            st.bar_chart(table)
        loc = data.get("lines_of_code")
        if loc:
            st.markdown("**Lines of kernel code (device code only)**")
            st.dataframe(pd.DataFrame([loc]).T.rename(columns={0: "kernel lines"}),
                         use_container_width=True)

# --- Tab 4: GPT integration -----------------------------------------------
with tab_gpt:
    st.subheader("GPT running on the custom CUDA attention")
    results_path = os.path.join(_INTEGRATION, "results.json")
    run_py = os.path.join(_INTEGRATION, "run_integration.py")

    with st.expander("▶ Run the integration (train → verify → generate)",
                     expanded=not os.path.exists(results_path)):
        col = st.columns(3)
        max_iters = col[0].number_input("Training iters", 50, 5000, 300, step=50)
        max_new = col[1].number_input("Tokens to generate", 50, 2000, 300, step=50)
        skip_cuda = col[2].checkbox("PyTorch only (skip CUDA build)", value=False,
                                    help="Tick this if the CUDA extension can't build in the "
                                         "environment Streamlit is running in.")
        st.caption("Runs with the same Python as this app. For the CUDA backend, launch "
                   "Streamlit from the .venv-triton (cu130) env in the VS x64 Native Tools shell.")
        if st.button("Run integration now", type="primary"):
            cmd = [sys.executable, run_py,
                   "--max-iters", str(int(max_iters)),
                   "--max-new-tokens", str(int(max_new))]
            if skip_cuda:
                cmd.append("--skip-cuda")
            with st.spinner("Training the GPT and generating text… this can take a minute."):
                proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
            st.session_state["integration_log"] = (proc.stdout or "") + "\n" + (proc.stderr or "")
            st.session_state["integration_rc"] = proc.returncode

    rc = st.session_state.get("integration_rc")
    if rc is not None:
        (st.success if rc == 0 else st.error)(
            "Finished — results below." if rc == 0 else f"Exited with code {rc} — see the run log.")
    if st.session_state.get("integration_log", "").strip():
        with st.expander("Run log"):
            st.code(st.session_state["integration_log"], language="text")

    if not os.path.exists(results_path):
        st.info("No results yet — run it above, or from a terminal: "
                "`python integration/run_integration.py`")
    else:
        data = load_json(results_path, os.path.getmtime(results_path))
        cfg = data.get("config", {})
        st.caption(f"Model: {data.get('params_millions')}M params · "
                   f"n_embd={cfg.get('n_embd')} n_head={cfg.get('n_head')} "
                   f"n_layer={cfg.get('n_layer')} block_size={cfg.get('block_size')}")

        c = st.columns(3)
        err = data.get("max_abs_error_vs_pytorch")
        c[0].metric("CUDA vs PyTorch max error", f"{err:.2e}" if err is not None else "—",
                    help="Max abs difference in logits between the two attention backends.")
        fwd = data.get("forward_ms", {})
        c[1].metric("Forward (PyTorch)", f"{fwd.get('pytorch', '—')} ms")
        c[2].metric("Forward (CUDA backend)", f"{fwd.get('cuda', '—')} ms")

        if err is not None and err < 1e-3:
            st.success("✅ The custom CUDA attention reproduces PyTorch attention "
                       "(outputs match to floating-point rounding).")
        if not data.get("cuda_backend_available", False) and data.get("cuda_backend_error"):
            st.warning(f"CUDA backend was not available: {data['cuda_backend_error']}")

        st.markdown(f"**Sample generated with the "
                    f"`{data.get('backend_used_for_generation')}` backend:**")
        st.code(data.get("sample_text", ""), language="text")

st.divider()
st.caption("Full write-up: docs/project_report.md · Commands: commands/commands.txt")
