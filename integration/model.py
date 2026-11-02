"""model.py - a small GPT (Karpathy-style) with a switchable attention backend.

Same architecture as gpt/v2.py, but written for integration:
  * import-safe: defining the model does NOT start training (no top-level script).
  * every attention head can run on the PyTorch backend OR our custom CUDA kernels,
    chosen at run time with  model.set_attention_backend('pytorch' | 'cuda').
  * attention is scaled by 1/sqrt(head_size) (the standard transformer scale), which
    is exactly what the CUDA kernel computes -- so both backends give the same output.

Training always uses the PyTorch backend (autograd); the CUDA kernels are forward-only.
Switch to the CUDA backend for inference / generation after the model is trained.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import the CUDA backend whether model.py is imported as a package or run from
# the integration/ directory as a loose script.
try:
    from .cuda_attention_backend import cuda_causal_attention
except ImportError:  # pragma: no cover - running as a loose script
    from cuda_attention_backend import cuda_causal_attention


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


class Head(nn.Module):
    """One head of self-attention with a selectable backend."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hs = cfg.head_size
        self.head_size = hs
        self.key = nn.Linear(cfg.n_embd, hs, bias=False)
        self.query = nn.Linear(cfg.n_embd, hs, bias=False)
        self.value = nn.Linear(cfg.n_embd, hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)
        self.backend = "pytorch"          # set via model.set_attention_backend(...)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)                 # (B, T, head_size)
        k = self.key(x)
        v = self.value(x)

        if self.backend == "cuda":
            # Our CUDA kernels: FP32, causal, scale 1/sqrt(head_size), forward-only.
            return cuda_causal_attention(q, k, v, optimized=True)

        # Standard PyTorch attention (also used for training).
        scale = self.head_size ** -0.5
        wei = (q @ k.transpose(-2, -1)) * scale          # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v                                    # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    # -- backend switch ------------------------------------------------------
    def set_attention_backend(self, backend: str):
        """Set every attention head to 'pytorch' or 'cuda'."""
        if backend not in ("pytorch", "cuda"):
            raise ValueError("backend must be 'pytorch' or 'cuda'")
        for module in self.modules():
            if isinstance(module, Head):
                module.backend = backend
        return self

    # -- forward / loss ------------------------------------------------------
    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        tok = self.token_embedding_table(idx)                       # (B, T, C)
        pos = self.position_embedding_table(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, vocab)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]                # crop to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                               # last step (B, vocab)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
