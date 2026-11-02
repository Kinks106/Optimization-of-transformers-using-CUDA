"""run_integration.py - end-to-end GPT integration demo (Phase 9 of the project).

This is the piece that connects the two halves of the project: the "GPT from
scratch" transformer and our custom CUDA attention kernels. It:

  1. Prints the GPU this project runs on (integration/gpu_info.py).
  2. Builds the small GPT (integration/model.py) and trains it with the PyTorch
     backend -- the CUDA kernels are forward-only, so training needs autograd.
  3. VERIFIES that the custom CUDA attention backend reproduces the PyTorch
     backend's output (max absolute error on the logits of a forward pass).
  4. Times one forward pass on each backend (a rough integration-level comparison;
     the headline kernel numbers come from the dedicated benchmarks).
  5. Generates text using the CUDA attention backend.
  6. Saves integration/results.json (GPU info + errors + sample text) for the UI.

Usage (from the repo root, in a shell where the CUDA extension can build -- see
triton_attention/SETUP.md):
    python integration/run_integration.py                 # quick defaults
    python integration/run_integration.py --max-iters 2000 --n-embd 384 --n-layer 6 --n-head 6
    python integration/run_integration.py --skip-cuda     # PyTorch backend only
"""

import argparse
import json
import os
import sys
import time

# Make the loose modules importable whether run as a script or a module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch                               # noqa: E402

from gpu_info import get_gpu_info, format_gpu_info     # noqa: E402
from model import GPTConfig, GPTLanguageModel          # noqa: E402

INPUT_TXT = os.path.join(_REPO, "gpt", "input.txt")
RESULTS_JSON = os.path.join(_HERE, "results.json")
WEIGHTS_PT = os.path.join(_HERE, "gpt_weights.pt")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(block_size, device, train_frac=0.9):
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split, batch_size):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    return dict(vocab_size=len(chars), encode=encode, decode=decode, get_batch=get_batch)


# ---------------------------------------------------------------------------
# Train (PyTorch backend)
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, get_batch, batch_size, eval_iters):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, batch_size)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, data, args, device):
    model.set_attention_backend("pytorch")          # training needs autograd
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    get_batch = data["get_batch"]
    print(f"\nTraining ({args.max_iters} iters, PyTorch backend) ...")
    for it in range(args.max_iters):
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, get_batch, args.batch_size, args.eval_iters)
            print(f"  step {it:>5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch("train", args.batch_size)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return losses


# ---------------------------------------------------------------------------
# Verify + time the CUDA backend
# ---------------------------------------------------------------------------
@torch.no_grad()
def verify_backends(model, data, block_size, device, batch_size=4):
    """Max abs error between PyTorch-backend and CUDA-backend logits (eval mode)."""
    model.eval()
    xb, _ = data["get_batch"]("val", batch_size)

    model.set_attention_backend("pytorch")
    logits_pt, _ = model(xb)
    model.set_attention_backend("cuda")
    logits_cuda, _ = model(xb)
    model.set_attention_backend("pytorch")

    return (logits_pt.double() - logits_cuda.double()).abs().max().item()


@torch.no_grad()
def time_forward(model, data, device, backend, iters=20, batch_size=8):
    model.eval()
    model.set_attention_backend(backend)
    xb, _ = data["get_batch"]("val", batch_size)
    for _ in range(3):                      # warm-up (also triggers the JIT build)
        model(xb)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model(xb)
    end.record()
    end.synchronize()
    model.set_attention_backend("pytorch")
    return start.elapsed_time(end) / iters  # ms per forward pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--skip-cuda", action="store_true", help="PyTorch backend only")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    # 1) GPU properties -----------------------------------------------------
    print(format_gpu_info())
    gpu_info = get_gpu_info()
    if device != "cuda":
        print("\nWARNING: no CUDA GPU; running the PyTorch backend on CPU. "
              "The CUDA attention backend needs a GPU.")
        args.skip_cuda = True

    # 2) Data + model -------------------------------------------------------
    data = load_data(args.block_size, device)
    cfg = GPTConfig(vocab_size=data["vocab_size"], block_size=args.block_size,
                    n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
                    dropout=args.dropout)
    model = GPTLanguageModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {n_params:.2f}M params | block_size={cfg.block_size} "
          f"n_embd={cfg.n_embd} n_head={cfg.n_head} n_layer={cfg.n_layer}")

    # 3) Train --------------------------------------------------------------
    final_losses = train(model, data, args, device)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, WEIGHTS_PT)

    results = {
        "gpu_info": gpu_info,
        "config": cfg.__dict__,
        "params_millions": round(n_params, 3),
        "final_loss": final_losses,
        "backend_used_for_generation": "pytorch",
        "cuda_backend_available": False,
        "cuda_backend_error": None,
        "max_abs_error_vs_pytorch": None,
        "forward_ms": {},
        "sample_text": "",
    }

    # 4) Verify + time + 5) generate ---------------------------------------
    gen_backend = "pytorch"
    if not args.skip_cuda and device == "cuda":
        from cuda_attention_backend import is_available
        ok, reason = is_available()
        results["cuda_backend_available"] = ok
        results["cuda_backend_error"] = reason or None
        if ok:
            print("\nVerifying CUDA attention backend against PyTorch backend ...")
            err = verify_backends(model, data, cfg.block_size, device)
            results["max_abs_error_vs_pytorch"] = err
            print(f"  max abs error (logits): {err:.3e}  "
                  f"({'MATCH' if err < 1e-3 else 'MISMATCH - check scaling'})")

            print("Timing one forward pass per backend ...")
            t_pt = time_forward(model, data, device, "pytorch")
            t_cu = time_forward(model, data, device, "cuda")
            results["forward_ms"] = {"pytorch": round(t_pt, 4), "cuda": round(t_cu, 4)}
            print(f"  forward pass: PyTorch {t_pt:.4f} ms | CUDA backend {t_cu:.4f} ms")
            gen_backend = "cuda"
        else:
            print(f"\nCUDA attention backend unavailable, using PyTorch for generation.\n  {reason}")

    print(f"\nGenerating {args.max_new_tokens} tokens with the '{gen_backend}' backend ...")
    model.eval()
    model.set_attention_backend(gen_backend)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_ids = model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()
    sample = data["decode"](sample_ids)
    results["backend_used_for_generation"] = gen_backend
    results["sample_text"] = sample
    print("\n" + "-" * 60 + "\n" + sample + "\n" + "-" * 60)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.relpath(RESULTS_JSON, _REPO)}  and  "
          f"{os.path.relpath(WEIGHTS_PT, _REPO)}")


if __name__ == "__main__":
    main()
