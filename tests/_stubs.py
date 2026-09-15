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


class StubInterventionRunner:
    """Deterministic stand-in for a causal LM, for the P3 suite.

    The stub has a single "readout" direction at one layer: the log-odds of
    answering ``Yes`` is the projection of that layer's hidden state onto the
    readout. That makes the expected control pattern exact rather than
    approximate -- an intervention at a different layer, or at a token other than
    the decision position, provably cannot move the endpoint, which is what the
    wrong-layer and wrong-token controls are supposed to demonstrate.

    Synthetic, and confined to this named stub module.
    """

    def __init__(self, dim: int = 16, n_layers: int = 8, readout_layer: int = 5,
                 seed: int = 0) -> None:
        import numpy as np

        self.dim = dim
        self.n_layers = n_layers
        self.readout_layer = readout_layer
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=dim)
        self.readout = vector / np.linalg.norm(vector)
        self.calls: list[dict] = []

    def hidden_at(self, prompt: str, layer: int, *, site: str = "block"):
        import hashlib

        import numpy as np

        digest = hashlib.sha256(f"{layer}|{prompt}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return rng.normal(size=self.dim)

    def choice_logprobs(self, prompt: str, options, *, intervention):
        import numpy as np

        hidden = self.hidden_at(prompt, self.readout_layer)
        # Only an intervention at the readout layer and the decision position can
        # influence the next-token readout.
        applies = (intervention.kind != "none"
                   and intervention.layer == self.readout_layer
                   and intervention.token_index == -1)
        if applies:
            if intervention.kind == "add":
                hidden = hidden + intervention.strength * np.asarray(intervention.vector)
            elif intervention.kind == "replace":
                hidden = np.asarray(intervention.vector, dtype=np.float64)
        logit = float(self.readout @ hidden)
        self.calls.append({"prompt": prompt, "kind": intervention.kind,
                           "layer": intervention.layer, "applied": applies})
        return {"Yes": logit, "No": -logit}
