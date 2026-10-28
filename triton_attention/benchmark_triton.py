"""
benchmark_triton.py - Step 4: fair benchmark of every GPU version, timed from Python.

Versions (all FP32, causal, single head, [N, d] row-major, d = 64):
    naive CUDA      kernels from cuda_attention/benchmark.cu, built as a PyTorch extension
    optimized CUDA  same file, shared-memory tiled Q@K^T
    Triton tiled    triton_attention/01_triton_tiled_qkt.py
    Triton fused    triton_attention/02_triton_fused_attention.py
    PyTorch SDPA    F.scaled_dot_product_attention(is_causal=True)
Separate FP16 table (not comparable with the FP32 CUDA code): Triton fused FP16, SDPA FP16.

Fairness rules applied to every version:
    * same Q, K, V tensors (U[-1, 1], fixed seed), already on the GPU
    * output / scores buffers allocated once, outside the timed region (like benchmark.cu)
    * every version is checked against SDPA before it is timed
    * same timer: CUDA events around a loop of --iters back-to-back calls, after
      --warmup untimed calls; repeated --rounds times; the median round is reported
    * two timing modes:
        eager      : normal Python calls (includes CPU launch overhead; this is what users see)
        CUDA graph : the same calls captured once and replayed (GPU time only, no Python
                     overhead), which matters for small N where launch overhead is large
    * no L2 flush between iterations (same as benchmark.cu)

Run from the repo root (for the CUDA extension: in an "x64 Native Tools Command Prompt
for VS 2022", with CUDA_PATH pointing at the CUDA toolkit):
    python triton_attention/benchmark_triton.py
    python triton_attention/benchmark_triton.py --skip-cuda   # Triton + SDPA only
Writes: triton_attention/results/benchmark_<gpu>.md and .json
"""

import argparse
import ast
import importlib
import json
import os
import re
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
tiled = importlib.import_module("01_triton_tiled_qkt")
fused = importlib.import_module("02_triton_fused_attention")

CUDA_BENCH_SRC = os.path.join(REPO, "cuda_attention", "benchmark.cu")
EXT_DIR = os.path.join(HERE, "cuda_ext")
EXT_KERNELS_SRC = os.path.join(EXT_DIR, "attention_kernels.cu")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ============================================================================
# CUDA extension
# ============================================================================
BUILD_HINTS = """
How to fix the most common Windows build errors:
  * "CUDA_HOME environment variable is not set" / nvcc not found:
        install the CUDA toolkit and set CUDA_PATH (e.g. C:\\Program Files\\NVIDIA GPU
        Computing Toolkit\\CUDA\\v13.3), then open a new terminal.
  * "Error checking compiler version for cl" / cl.exe not found / missing headers (stddef.h):
        run from the "x64 Native Tools Command Prompt for VS 2022" (or run vcvars64.bat first).
  * "The detected CUDA version (X) mismatches the version that was used to compile PyTorch (Y)":
        torch and nvcc must have the same CUDA MAJOR version: use torch from the cu130 index
        with CUDA 13.x (see triton_attention/SETUP.md).
  * "ninja is required": pip install ninja
  * Stale build after changing versions: delete %LOCALAPPDATA%\\torch_extensions and retry.
To run only the Triton and SDPA versions, add --skip-cuda.
"""


def build_cuda_ext(verbose: bool):
    from torch.utils.cpp_extension import load

    t0 = time.time()
    try:
        ext = load(
            name="attention_cuda_ext",
            sources=[os.path.join(EXT_DIR, "attention_ext.cpp"), EXT_KERNELS_SRC],
            extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
            extra_cuda_cflags=["-O3"],  # no --use_fast_math: keep expf accurate, as in benchmark.cu
            verbose=verbose,
        )
    except Exception as exc:  # report the real error, then stop
        print("\nERROR: building the CUDA extension failed:\n")
        print(f"    {type(exc).__name__}: {exc}")
        print(BUILD_HINTS)
        sys.exit(2)
    print(f"CUDA extension built/loaded in {time.time() - t0:.1f} s")
    return ext


# ============================================================================
# Timing
# ============================================================================
def _rounds(step, iters, rounds):
    times = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return times


def time_eager(fn, warmup, iters, rounds):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return _rounds(fn, iters, rounds)


def time_graph(fn, warmup, iters, rounds):
    # Warm up on a side stream before capture (recommended by the PyTorch CUDA-graph docs).
    # Triton autotuning also finishes here, so it is never captured or timed.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    return _rounds(graph.replay, iters, rounds)


# ============================================================================
# Lines of kernel code
# ============================================================================
def _cuda_kernel_text(path, name):
    src = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    m = re.search(r"__global__\s+void\s+" + name + r"\s*\(", src)
    if not m:
        raise ValueError(f"kernel {name} not found in {path}")
    i = src.index("{", m.end())
    depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start():j + 1]
    raise ValueError(f"unbalanced braces in {name}")


def _count_c_code_lines(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return sum(1 for ln in text.split("\n") if ln.split("//")[0].strip())


def _triton_kernel_lines(path, names):
    src = open(path, encoding="utf-8").read()
    lines = src.split("\n")
    total = 0
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body = lines[node.lineno - 1:node.end_lineno]
            total += sum(1 for ln in body if ln.strip() and not ln.strip().startswith("#"))
    return total


def lines_of_code():
    cuda = lambda names: sum(_count_c_code_lines(_cuda_kernel_text(CUDA_BENCH_SRC, n)) for n in names)
    rows = {
        "naive CUDA": (cuda(["naiveScoresKernel", "softmaxKernel", "outputKernel"]), 3,
                       "cuda_attention/benchmark.cu"),
        "optimized CUDA": (cuda(["optimizedScoresKernel", "softmaxKernel", "outputKernel"]), 3,
                           "cuda_attention/benchmark.cu"),
        "Triton tiled": (_triton_kernel_lines(tiled.__file__, {"qkt_scores_kernel", "softmax_rows_kernel",
                                                                "pv_kernel"}), 3,
                         "triton_attention/01_triton_tiled_qkt.py"),
        "Triton fused": (_triton_kernel_lines(fused.__file__, {"fused_causal_attention_kernel"}), 1,
                         "triton_attention/02_triton_fused_attention.py"),
    }
    # Fairness check: the extension must contain exactly the kernels we wrote for the project.
    same = all(
        _cuda_kernel_text(CUDA_BENCH_SRC, n).strip() == _cuda_kernel_text(EXT_KERNELS_SRC, n).strip()
        for n in ["naiveScoresKernel", "optimizedScoresKernel", "softmaxKernel", "outputKernel"]
    )
    return rows, same


# ============================================================================
# Benchmark
# ============================================================================
def sdpa(q4, k4, v4):
    return F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)


def max_err(a, b):
    return (a.double() - b.double()).abs().max().item()


def run(args):
    ext = None if args.skip_cuda else build_cuda_ext(args.verbose_build)
    modes = [("eager", time_eager)] + ([] if args.no_graph else [("graph", time_graph)])

    results = []  # one dict per (N, version, mode)
    errors = []   # (N, version, max abs err vs SDPA FP32)
    stages = []   # (N, stage, version, mode, median ms)

    def record(N, version, group, fn, mode, timer):
        t = timer(fn, args.warmup, args.iters, args.rounds)
        results.append(dict(seq_len=N, version=version, group=group, mode=mode,
                            median_ms=statistics.median(t), min_ms=min(t), max_ms=max(t), rounds_ms=t))

    for N in args.seq_lens:
        d = args.head_dim
        g = torch.Generator(device="cuda").manual_seed(args.seed + N)
        q, k, v = (torch.rand(N, d, device="cuda", generator=g) * 2 - 1 for _ in range(3))
        q4, k4, v4 = q[None, None], k[None, None], v[None, None]
        q16, k16, v16 = q.half(), k.half(), v.half()
        q16_4, k16_4, v16_4 = q16[None, None], k16[None, None], v16[None, None]
        new = lambda *shape, dt=torch.float32: torch.empty(*shape, device="cuda", dtype=dt)

        fp32 = {}
        if ext is not None:
            s1, o1, s2, o2 = new(N, N), new(N, d), new(N, N), new(N, d)
            fp32["naive CUDA"] = lambda: ext.attention(q, k, v, s1, o1, False)
            fp32["optimized CUDA"] = lambda: ext.attention(q, k, v, s2, o2, True)
        s3, o3, o4 = new(N, N), new(N, d), new(N, d)
        fp32["Triton tiled"] = lambda: tiled.triton_tiled_attention(q, k, v, s3, o3)
        fp32["Triton fused"] = lambda: fused.triton_fused_attention(q, k, v, o4)
        fp32["PyTorch SDPA"] = lambda: sdpa(q4, k4, v4)

        o5 = new(N, d, dt=torch.float16)
        fp16 = {
            "Triton fused FP16": lambda: fused.triton_fused_attention(q16, k16, v16, o5),
            "PyTorch SDPA FP16": lambda: sdpa(q16_4, k16_4, v16_4),
        }

        # Correctness gate: never time a version that gives wrong answers.
        ref = sdpa(q4, k4, v4)[0, 0]
        for name, fn in {**fp32, **fp16}.items():
            out = fn()
            out = out[0, 0] if out.dim() == 4 else out
            torch.cuda.synchronize()
            err = max_err(out, ref)
            errors.append((N, name, err))
            tol = 1e-2 if "FP16" in name else 1e-5
            if not err < tol:
                raise RuntimeError(f"{name} at N={N} is wrong: max abs err {err:.3e} (tolerance {tol})")

        for mode, timer in modes:
            for name, fn in fp32.items():
                record(N, name, "fp32", fn, mode, timer)
            for name, fn in fp16.items():
                record(N, name, "fp16", fn, mode, timer)

        # Per-stage times. Re-running a stage on its own output does the same
        # amount of work every call, so the timing is valid even though values change.
        st = {}
        if ext is not None:
            st[("1. Q@K^T + mask", "naive CUDA")] = lambda: ext.scores(q, k, s1, False)
            st[("1. Q@K^T + mask", "optimized CUDA")] = lambda: ext.scores(q, k, s2, True)
            st[("2. softmax", "CUDA (both)")] = lambda: ext.softmax(s2)
            st[("3. P@V", "CUDA (both)")] = lambda: ext.output(s2, v, o2)
        st[("1. Q@K^T + mask", "Triton tiled")] = lambda: tiled.qkt_scores(q, k, s3)
        st[("2. softmax", "Triton tiled")] = lambda: tiled.softmax_rows(s3)
        st[("3. P@V", "Triton tiled")] = lambda: tiled.pv(s3, v, o3)
        for mode, timer in modes:
            for (stage, name), fn in st.items():
                t = timer(fn, args.warmup, args.iters, args.rounds)
                stages.append((N, stage, name, mode, statistics.median(t)))
        print(f"  N = {N:>5} done")

    return results, errors, stages


# ============================================================================
# Report
# ============================================================================
def fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def report(args, results, errors, stages, loc, kernels_same):
    gpu = torch.cuda.get_device_name(0)
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    get = lambda N, ver, mode: next((r["median_ms"] for r in results
                                     if r["seq_len"] == N and r["version"] == ver and r["mode"] == mode), None)
    spread = lambda N, mode: max(((r["max_ms"] - r["min_ms"]) / r["median_ms"] * 100
                                  for r in results if r["seq_len"] == N and r["mode"] == mode), default=0)

    fp32_cols = ["naive CUDA", "optimized CUDA", "Triton tiled", "Triton fused", "PyTorch SDPA"]
    fp16_cols = ["Triton fused FP16", "PyTorch SDPA FP16"]
    mode_names = {"eager": "eager (Python calls, includes launch overhead)",
                  "graph": "CUDA graph replay (GPU time only)"}

    out = ["<!-- generated by benchmark_triton.py -->",
           f"**GPU:** {gpu} ({cap}) | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
           f"triton {triton.__version__} | d = {args.head_dim} | FP32 | warmup {args.warmup}, "
           f"{args.iters} iters x {args.rounds} rounds, median round reported", ""]

    for mode in sorted({r["mode"] for r in results}, key=["eager", "graph"].index):
        out += [f"### Time per attention call (ms), {mode_names[mode]}", "",
                "| Seq len | " + " | ".join(fp32_cols) + " | spread |",
                "|---:|" + "---:|" * len(fp32_cols) + "---:|"]
        for N in args.seq_lens:
            out.append(f"| {N} | " + " | ".join(fmt(get(N, c, mode)) for c in fp32_cols)
                       + f" | {spread(N, mode):.0f}% |")
        out.append("")
        for base in ("naive CUDA", "optimized CUDA"):
            if get(args.seq_lens[0], base, mode) is None:
                continue
            others = [c for c in fp32_cols if c != base]
            out += [f"Speedup vs **{base}** ({mode}; >1 = faster than {base}):", "",
                    "| Seq len | " + " | ".join(others) + " |", "|---:|" + "---:|" * len(others)]
            for N in args.seq_lens:
                b = get(N, base, mode)
                out.append(f"| {N} | " + " | ".join(
                    f"{b / get(N, c, mode):.2f}x" if get(N, c, mode) else "n/a" for c in others) + " |")
            out.append("")
        out += [f"Separate FP16 table ({mode}; not comparable with the FP32 CUDA code):", "",
                "| Seq len | " + " | ".join(fp16_cols) + " |", "|---:|" + "---:|" * len(fp16_cols)]
        for N in args.seq_lens:
            out.append(f"| {N} | " + " | ".join(fmt(get(N, c, mode)) for c in fp16_cols) + " |")
        out.append("")

    stage_cols = sorted({s[2] for s in stages}, key=_stage_order)
    for mode in sorted({s[3] for s in stages}, key=["eager", "graph"].index):
        out += [f"### Per-stage time (ms), {mode_names[mode]}", "",
                "| Seq len | Stage | " + " | ".join(stage_cols) + " |",
                "|---:|---|" + "---:|" * len(stage_cols)]
        for N in args.seq_lens:
            for stage in sorted({s[1] for s in stages}):
                vals = {s[2]: s[4] for s in stages if s[0] == N and s[1] == stage and s[3] == mode}
                out.append(f"| {N} | {stage} | " + " | ".join(fmt(vals.get(c)) if c in vals else "—"
                                                             for c in stage_cols) + " |")
        out.append("")

    out += ["### Max abs error vs PyTorch SDPA FP32 (checked before timing)", "",
            "| Seq len | " + " | ".join(fp32_cols[:-1] + fp16_cols) + " |",
            "|---:|" + "---:|" * (len(fp32_cols) - 1 + len(fp16_cols))]
    for N in args.seq_lens:
        e = {name: err for n, name, err in errors if n == N}
        out.append(f"| {N} | " + " | ".join(f"{e[c]:.2e}" if c in e else "n/a"
                                            for c in fp32_cols[:-1] + fp16_cols) + " |")
    out.append("")

    out += ["### Lines of kernel code (device code only; blank and comment lines excluded)", "",
            "| Implementation | Kernel lines | Kernels | Source |", "|---|---:|---:|---|"]
    for name, (n, kernels, src) in loc.items():
        out.append(f"| {name} | {n} | {kernels} | `{src}` |")
    out += ["| PyTorch SDPA | — (library call) | — | — |", "",
            f"CUDA extension kernels identical to `cuda_attention/benchmark.cu`: "
            f"**{'yes' if kernels_same else 'NO'}**"]
    return "\n".join(out)


def _stage_order(name):
    order = ["naive CUDA", "optimized CUDA", "CUDA (both)", "Triton tiled"]
    return order.index(name) if name in order else len(order)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-cuda", action="store_true", help="do not build/run the CUDA extension")
    p.add_argument("--no-graph", action="store_true", help="skip the CUDA-graph timing mode")
    p.add_argument("--verbose-build", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available in this PyTorch build (see triton_attention/SETUP.md).")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    loc, kernels_same = lines_of_code()
    if not kernels_same:
        print("WARNING: CUDA extension kernels differ from cuda_attention/benchmark.cu")

    results, errors, stages = run(args)
    text = report(args, results, errors, stages, loc, kernels_same)
    print("\n" + text)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", torch.cuda.get_device_name(0).lower()).strip("_")
    base = os.path.join(HERE, "results", f"benchmark_{slug}")
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
                       args=vars(args), results=results,
                       errors=[dict(seq_len=n, version=v, max_abs_err=e) for n, v, e in errors],
                       stages=[dict(seq_len=n, stage=s, version=v, mode=m, median_ms=t)
                               for n, s, v, m, t in stages],
                       lines_of_code={k: v[0] for k, v in loc.items()}), fh, indent=1)
    print(f"\nSaved: {os.path.relpath(base)}.md and .json")


if __name__ == "__main__":
    main()
