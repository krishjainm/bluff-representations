"""Explicitly named test stubs.

Not a test module.  The hidden states produced here are synthetic and exist only
so that pipeline logic can be exercised without downloading model weights or
contacting a network.  Research-integrity rule: synthetic activations are
permitted only in named tests/demos and must never reach a real research run.
"""
from __future__ import annotations

from typing import Sequence

import torch


class StubHiddenStateProvider:
    """Deterministic word-level stand-in for a tokenizer plus causal LM.

    Token ids are content-derived, so a test can assert exactly which text
    reached the model.  Each hidden state encodes its own token id in channel 0
    and its sequence position in channel 1, which lets a test verify that the
    recorded ``token_index`` is the position actually read.
    """

    def __init__(self, n_layers: int = 4, hidden: int = 3) -> None:
        self.n_layers = n_layers
        self.hidden = hidden
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        # Stable across processes, unlike hash(); PYTHONHASHSEED must not matter.
        return [sum(ord(c) * (i + 1) for i, c in enumerate(tok)) % 1000 + 1
                for tok in text.split()]

    def hidden_states(self, token_ids: Sequence[int], site: str) -> torch.Tensor:
        ids = torch.tensor(list(token_ids), dtype=torch.float32)
        positions = torch.arange(len(ids), dtype=torch.float32)
        base = torch.stack([ids, positions, torch.ones_like(ids)], dim=-1)
        return torch.stack([base + layer for layer in range(self.n_layers)], dim=0)
