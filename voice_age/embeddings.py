"""
voice_age/embeddings.py - frozen pretrained speech encoder -> fixed-size embeddings.

    16 kHz float32 clip -> encoder (frozen, eval, no gradients) -> hidden states of
    every layer -> mean over time (padding masked out) -> [layers + 1, hidden]

The encoder is never trained or modified. It is a separate instance from the
voice-emotion model (superb/hubert-base-superb-er) and from the gender
pipeline's encoder; nothing is shared at runtime.
"""

import threading

import numpy as np

from . import config


def pick_device(preferred=config.AGE_DEVICE):
    import torch
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SpeechEncoder:
    def __init__(self, name=config.AGE_ENCODER, keep_layers=None, device=None):
        """keep_layers: keep only the first N transformer layers (inference needs only up to
        the chosen layer; hidden_states[k] depends only on layers < k)."""
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        self.name = name
        self.device = device or pick_device()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        if keep_layers is not None:
            model.encoder.layers = model.encoder.layers[:max(1, int(keep_layers))]
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.device).eval()
        self.n_layers = len(model.encoder.layers)
        self.hidden = model.config.hidden_size
        self._lock = threading.Lock()
        self._torch = torch

    def embed(self, clips):
        """list of 1-D float32 16 kHz arrays -> np.float32 [B, n_layers + 1, hidden]."""
        torch = self._torch
        inputs = self.feature_extractor(list(clips), sampling_rate=config.SAMPLE_RATE,
                                        return_tensors="pt", padding=True)
        x = inputs.input_values.to(self.device)
        with self._lock, torch.inference_mode():
            out = self.model(x, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=1)                          # B, L+1, T, H
            lengths = self.model._get_feat_extract_output_lengths(
                torch.tensor([len(c) for c in clips])).to(self.device)
            steps = torch.arange(hs.shape[2], device=self.device)
            mask = (steps[None, :] < lengths[:, None]).to(hs.dtype)             # ignore padding frames
            pooled = (hs * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]
        return pooled.float().cpu().numpy()

    def embed_batched(self, clips, batch_seconds=config.EMBED_BATCH_SECONDS):
        """Sort by length and batch by total audio seconds (little padding, bounded memory)."""
        order = np.argsort([len(c) for c in clips])
        out = [None] * len(clips)
        batch, total = [], 0.0
        for i in order:
            batch.append(i)
            total += len(clips[i]) / config.SAMPLE_RATE
            if total >= batch_seconds:
                for j, e in zip(batch, self.embed([clips[k] for k in batch])):
                    out[j] = e
                batch, total = [], 0.0
        if batch:
            for j, e in zip(batch, self.embed([clips[k] for k in batch])):
                out[j] = e
        return np.stack(out)
