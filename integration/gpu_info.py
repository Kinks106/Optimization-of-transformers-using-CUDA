"""gpu_info.py - report the GPU this project is running on.

This gives one concrete, live instance of the hardware our custom CUDA attention
kernels execute on. It is used by the GPT integration demo (run_integration.py)
and by the Streamlit UI (ui/app.py).

It works with or without PyTorch installed:
  * PyTorch present  -> rich info straight from the CUDA runtime
                        (compute capability, SM count, memory, CUDA version)
  * PyTorch absent   -> falls back to parsing `nvidia-smi`

Run directly to print a table:
    python integration/gpu_info.py
"""

from __future__ import annotations

import shutil
import subprocess


# ---------------------------------------------------------------------------
# nvidia-smi (no PyTorch needed) - also used for live stats like temperature
# ---------------------------------------------------------------------------
def query_nvidia_smi() -> list[dict] | None:
    """Return a list (one dict per GPU) from nvidia-smi, or None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    fields = [
        "name", "driver_version", "memory.total", "memory.used", "memory.free",
        "compute_cap", "temperature.gpu", "utilization.gpu",
        "power.draw", "power.limit",
    ]
    try:
        out = subprocess.run(
            [exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except Exception:
        return None

    rows = []
    for line in out.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows or None


def query_cuda_runtime_version() -> str | None:
    """CUDA toolkit version from `nvcc --version`, if nvcc is on PATH."""
    exe = shutil.which("nvcc")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=15, check=True).stdout
    except Exception:
        return None
    for token in out.replace(",", " ").split():
        if token.startswith("V") and token[1:2].isdigit():
            return token[1:]           # e.g. "13.3.73"
    return None


# ---------------------------------------------------------------------------
# Combined info (prefers PyTorch, enriches with nvidia-smi)
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict:
    """Collect GPU properties into a plain dict (JSON-serialisable)."""
    info: dict = {
        "available": False,
        "source": None,
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "nvcc_version": query_cuda_runtime_version(),
        "gpus": [],
    }

    # 1) PyTorch path (richest) -------------------------------------------------
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["available"] = True
            info["source"] = "torch"
            info["cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "index": i,
                    "name": p.name,
                    "compute_capability": f"sm_{p.major}{p.minor}",
                    "total_memory_gb": round(p.total_memory / 1024 ** 3, 2),
                    "multi_processor_count": p.multi_processor_count,
                    # Optional attributes (present on newer torch builds):
                    "l2_cache_mb": round(getattr(p, "L2_cache_size", 0) / 1024 ** 2, 2)
                    if getattr(p, "L2_cache_size", 0) else None,
                    "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", None),
                })
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"

    # 2) nvidia-smi enrichment / fallback --------------------------------------
    smi = query_nvidia_smi()
    if smi:
        info["nvidia_smi"] = smi
        # Attach live stats (driver, temperature, utilisation) to torch GPUs by index
        for i, row in enumerate(smi):
            live = {
                "driver_version": row.get("driver_version"),
                "temperature_c": row.get("temperature.gpu"),
                "utilization_pct": row.get("utilization.gpu"),
                "memory_used_mb": row.get("memory.used"),
                "memory_free_mb": row.get("memory.free"),
                "power_draw_w": row.get("power.draw"),
                "power_limit_w": row.get("power.limit"),
            }
            if i < len(info["gpus"]):
                info["gpus"][i].update(live)
            else:
                info["gpus"].append({
                    "index": i,
                    "name": row.get("name"),
                    "compute_capability": "sm_" + row.get("compute_cap", "").replace(".", ""),
                    "total_memory_gb": round(float(row["memory.total"]) / 1024, 2)
                    if row.get("memory.total") else None,
                    **live,
                })
        if not info["available"]:
            info["available"] = True
            info["source"] = "nvidia-smi"

    return info


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def format_gpu_info(info: dict | None = None) -> str:
    info = info or get_gpu_info()
    lines = ["=" * 60, " GPU PROPERTIES (hardware running the CUDA kernels)", "=" * 60]

    if not info["available"]:
        lines.append("No CUDA-capable GPU detected (or no driver / nvidia-smi).")
        if info.get("torch_error"):
            lines.append(f"torch note: {info['torch_error']}")
        return "\n".join(lines)

    lines.append(f"Info source      : {info['source']}")
    if info.get("torch_version"):
        lines.append(f"PyTorch          : {info['torch_version']} "
                     f"(CUDA {info.get('cuda_version')})")
    if info.get("nvcc_version"):
        lines.append(f"nvcc (toolkit)   : {info['nvcc_version']}")

    for g in info["gpus"]:
        lines.append("-" * 60)
        lines.append(f"GPU {g.get('index', 0)}          : {g.get('name')}")
        lines.append(f"Compute capable  : {g.get('compute_capability')}")
        if g.get("total_memory_gb") is not None:
            lines.append(f"Total memory     : {g.get('total_memory_gb')} GB")
        if g.get("multi_processor_count") is not None:
            lines.append(f"SM (cores groups): {g.get('multi_processor_count')} streaming multiprocessors")
        if g.get("max_threads_per_sm"):
            lines.append(f"Threads / SM     : {g.get('max_threads_per_sm')}")
        if g.get("l2_cache_mb"):
            lines.append(f"L2 cache         : {g.get('l2_cache_mb')} MB")
        if g.get("driver_version"):
            lines.append(f"Driver           : {g.get('driver_version')}")
        if g.get("memory_used_mb"):
            lines.append(f"Memory in use    : {g.get('memory_used_mb')} / "
                         f"{int(float(g.get('memory_used_mb')) + float(g.get('memory_free_mb')))} MB")
        if g.get("temperature_c"):
            lines.append(f"Temperature      : {g.get('temperature_c')} C")
        if g.get("utilization_pct"):
            lines.append(f"Utilisation      : {g.get('utilization_pct')} %")
        if g.get("power_draw_w") and g.get("power_limit_w"):
            lines.append(f"Power            : {g.get('power_draw_w')} / {g.get('power_limit_w')} W")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_gpu_info())
