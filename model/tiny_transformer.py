# model/tiny_transformer.py

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim muss durch num_heads teilbar sein"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (Batch, Time, EmbedDim)
        mask: (1, 1, Time, Time) mit 0/1 für erlaubte Positionen
        """
        B, T, C = x.shape

        # Linearprojektion
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # In Köpfe aufteilen
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = attn_probs @ v  # (B, H, T, D)

        # Köpfe wieder zusammenführen
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output-Projektion
        out = self.out_proj(attn_output)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x, mask):
        # Pre-LN Architektur
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """
    Kleines Decoder-only Transformer Language Model.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        """
        idx: (Batch, Time) mit Token-IDs
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        # Token- & Positions-Embeddings
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, C)

        # Causal Mask (nur in die Vergangenheit gucken)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(1)
        # Shape: (1, 1, T, T)

        # Transformer-Blöcke
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, Vocab)
        return logits