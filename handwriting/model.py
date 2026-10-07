"""
handwriting/model.py - small attention-based Transformer for pen trajectories.

    (B, T, F) features + (B, T) validity mask
        -> Linear + LayerNorm            input projection to d_model
        -> + sinusoidal positions        order of the samples in time
        -> TransformerEncoder            pre-norm, padded positions masked out
        -> attention pooling             learned weight per time step (softmax over
                                         valid steps only) -> one vector
        -> one Linear head per output    logits per configured label head

Heads are independent classifiers, one per EMOTHAW target (anxiety, depression,
stress), each over normal / elevated (see config.EMOTHAW_HEADS).
"""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class AttentionPooling(nn.Module):
    """score_t = v . tanh(W h_t); weights = softmax over valid t; output = sum_t w_t h_t."""

    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or d_model
        self.proj = nn.Linear(d_model, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, h, mask):
        scores = self.score(torch.tanh(self.proj(h))).squeeze(-1)          # B, T
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), h).squeeze(1), weights      # B, D ; B, T


class HandwritingTransformer(nn.Module):
    def __init__(self, n_features, heads, d_model=64, n_heads=4, n_layers=3,
                 dim_feedforward=128, dropout=0.2, max_len=4096):
        """heads: list of {"name": str, "classes": [str, ...]} (extra keys ignored)."""
        super().__init__()
        self.head_names = [h["name"] for h in heads]
        self.input = nn.Sequential(nn.Linear(n_features, d_model), nn.LayerNorm(d_model))
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPooling(d_model)
        self.heads = nn.ModuleDict({
            h["name"]: nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, len(h["classes"])))
            for h in heads})

    def forward(self, x, mask, return_attention=False):
        """x: (B, T, F) float; mask: (B, T) bool, True = real sample. -> {head: (B, C) logits}."""
        h = self.dropout(self.pos(self.input(x)))
        h = self.encoder(h, src_key_padding_mask=~mask)
        pooled, weights = self.pool(self.norm(h), mask)
        logits = {name: head(pooled) for name, head in self.heads.items()}
        if return_attention:
            return logits, weights
        return logits


def build_model(n_features, heads, hparams):
    return HandwritingTransformer(n_features, heads, **hparams)


def pad_batch(matrices):
    """list of (T_i, F) float32 arrays -> x (B, T, F), mask (B, T)."""
    lengths = [m.shape[0] for m in matrices]
    T, F = max(lengths), matrices[0].shape[1]
    x = torch.zeros(len(matrices), T, F)
    mask = torch.zeros(len(matrices), T, dtype=torch.bool)
    for i, m in enumerate(matrices):
        x[i, : m.shape[0]] = torch.from_numpy(m)
        mask[i, : m.shape[0]] = True
    return x, mask
