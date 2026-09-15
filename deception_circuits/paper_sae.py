"""One defensible unsupervised sparse autoencoder path for the strict paper.

Built after auditing the legacy module (see `docs/SAE_AUDIT.md`), whose saved
"best autoencoder" was in fact a label-supervised classifier-autoencoder selected
by classification accuracy. Nothing here uses labels to *train* or to select the
dictionary; labels enter only to **rank** already-frozen features on the
train/validation partitions, after which the selected features are scored once on
test.

Design choices and why:

- TopK activation by default. L0 is then exact and there is no L1 shrinkage for
  the model to trade against decoder norm growth.
- Decoder columns renormalised to unit norm after every optimizer step, so
  feature magnitudes are comparable and the sparsity objective is not gameable.
- Genuine weight tying when requested: the decoder uses the encoder parameter
  transposed *inside* the autograd graph. The legacy module copied weights
  in-place under ``no_grad`` during ``forward``, which both broke the gradient
  path and mutated parameters at inference.
- Input normalization statistics are fitted on the training partition only and
  stored with the model.
- :meth:`SparseAutoencoder.freeze` is called before any held-out evaluation, and
  a test asserts that evaluating a frozen SAE changes no weight.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .paper import PaperConfig, ResearchIntegrityError, compute_binary_metrics

ACTIVATIONS = ("topk", "relu")


@dataclass(frozen=True)
class SAEConfig:
    """Every choice that changes the learned dictionary."""

    n_features: int
    activation: str = "topk"
    # Required for topk: the exact number of active features per example.
    k: int | None = None
    # Only used by relu; summed over features, meaned over batch, so the
    # effective strength does not depend on the dictionary width.
    l1_coefficient: float = 1e-3
    tied_weights: bool = False
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    seed: int = 0
    normalize_inputs: bool = True
    # Features whose activation frequency falls below this are counted dead.
    dead_feature_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.n_features < 2:
            raise ResearchIntegrityError("n_features must be >= 2 and must be stated explicitly")
        if self.activation not in ACTIVATIONS:
            raise ResearchIntegrityError(f"activation must be one of {list(ACTIVATIONS)}")
        if self.activation == "topk":
            if self.k is None:
                raise ResearchIntegrityError("topk activation requires an explicit k")
            if not 1 <= self.k <= self.n_features:
                raise ResearchIntegrityError(f"k must be in [1, n_features]; got {self.k}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ResearchIntegrityError("epochs and batch_size must be >= 1")
        if self.l1_coefficient < 0:
            raise ResearchIntegrityError("l1_coefficient must be non-negative")

    def expansion_factor(self, input_dim: int) -> float:
        return float(self.n_features) / float(input_dim)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SparseAutoencoder(nn.Module):
    """Unsupervised SAE with a unit-norm decoder and optional genuine tying."""

    def __init__(self, input_dim: int, config: SAEConfig) -> None:
        super().__init__()
        if input_dim < 1:
            raise ResearchIntegrityError("input_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        scale = 1.0 / np.sqrt(input_dim)
        self.encoder_weight = nn.Parameter(
            torch.randn(config.n_features, input_dim, generator=generator) * scale)
        self.encoder_bias = nn.Parameter(torch.zeros(config.n_features))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        if config.tied_weights:
            # No separate decoder parameter: decode() transposes the encoder
            # weight inside the graph, which is what tying actually means.
            self.decoder_weight = None
        else:
            self.decoder_weight = nn.Parameter(
                torch.randn(input_dim, config.n_features, generator=generator) * scale)
        self.normalize_decoder_()

    @property
    def n_features(self) -> int:
        return self.config.n_features

    def decoder_matrix(self) -> torch.Tensor:
        """``[input_dim, n_features]``; a view of the encoder when tied."""
        if self.decoder_weight is None:
            return self.encoder_weight.t()
        return self.decoder_weight

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Renormalise decoder columns to unit norm.

        Called after every optimizer step. When weights are tied this normalises
        the encoder rows, which is the same dictionary.
        """
        target = self.encoder_weight if self.decoder_weight is None else self.decoder_weight
        dim = 1 if self.decoder_weight is None else 0
        norms = target.norm(dim=dim, keepdim=True).clamp(min=1e-8)
        target.div_(norms)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pre_activation, code)``; the code is always non-negative."""
        pre = F.linear(x, self.encoder_weight, self.encoder_bias)
        if self.config.activation == "topk":
            # ReLU first so the code is non-negative, then keep the k largest.
            rectified = F.relu(pre)
            k = int(self.config.k)
            values, indices = torch.topk(rectified, k, dim=-1)
            code = torch.zeros_like(rectified).scatter_(-1, indices, values)
        else:
            code = F.relu(pre)
        return pre, code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        # decoder_matrix is [input_dim, n_features]; transpose to right-multiply the code.
        return code @ self.decoder_matrix().t() + self.decoder_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre, code = self.encode(x)
        return self.decode(code), code, pre

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        reconstruction, code, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x)
        # Summed over features, meaned over batch: width-independent.
        l1 = code.abs().sum(dim=-1).mean()
        total = mse + (self.config.l1_coefficient * l1
                       if self.config.activation == "relu" else 0.0)
        return total, {"mse": float(mse.detach()), "l1": float(l1.detach()),
                       "total": float(total.detach())}

    def freeze(self) -> "SparseAutoencoder":
        """Put the SAE beyond modification before any held-out evaluation."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def decoder_directions(self) -> np.ndarray:
        """Unit-norm feature directions as ``[n_features, input_dim]``."""
        matrix = self.decoder_matrix().detach().cpu().numpy().T
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-8, None)

    def parameter_fingerprint(self) -> str:
        """Hash of every parameter, so a frozen-SAE test can prove immutability."""
        import hashlib

        digest = hashlib.sha256()
        for name, parameter in sorted(self.named_parameters()):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Input normalization
# --------------------------------------------------------------------------- #

def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a device string, refusing a silent fallback to CPU.

    A run that asked for cuda and quietly got cpu would take hours instead of
    minutes and look like it had simply been slow, so an unavailable device is an
    error rather than a downgrade.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ResearchIntegrityError(
            "device='cuda' was requested but torch reports no CUDA device. Refusing to fall "
            "back to CPU silently: an SAE of this width trains for hours on CPU.")
    return resolved


def _module_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def fit_input_normalizer(x_train: np.ndarray) -> dict[str, Any]:
    """Centre and rescale so ``E||x|| == sqrt(d)``, fitted on train only."""
    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.ndim != 2:
        raise ResearchIntegrityError("Normalizer expects [samples, features]")
    mean = x_train.mean(axis=0)
    centred = x_train - mean
    average_norm = float(np.linalg.norm(centred, axis=1).mean())
    if average_norm < 1e-12:
        raise ResearchIntegrityError("Training activations are constant; nothing to encode")
    scale = float(np.sqrt(x_train.shape[1]) / average_norm)
    return {"mean": mean.tolist(), "scale": scale, "n_train": int(len(x_train)),
            "fitted_on": "train partition only"}


def apply_input_normalizer(x: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    if stats is None:
        return np.asarray(x, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    return ((np.asarray(x, dtype=np.float64) - mean) * float(stats["scale"])).astype(np.float32)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_sae(
    x_train: np.ndarray, x_validation: np.ndarray, config: SAEConfig,
    *, device: str | torch.device = "cpu",
) -> tuple[SparseAutoencoder, dict[str, Any]]:
    """Train unsupervised on train, early-select the epoch on validation MSE.

    No labels are used anywhere in this function. The returned model is the
    best-validation checkpoint, restored and frozen.
    """
    x_train = np.asarray(x_train)
    if x_train.ndim != 2:
        raise ResearchIntegrityError("SAE training expects [samples, features]")
    if len(x_train) < config.batch_size:
        raise ResearchIntegrityError(
            f"{len(x_train)} training rows is fewer than batch_size={config.batch_size}")
    target = resolve_device(device)
    normalizer = fit_input_normalizer(x_train) if config.normalize_inputs else None
    train_tensor = torch.from_numpy(apply_input_normalizer(x_train, normalizer)).to(target)
    validation_tensor = torch.from_numpy(apply_input_normalizer(x_validation, normalizer)).to(target)

    torch.manual_seed(config.seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = SparseAutoencoder(x_train.shape[1], config).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    best_epoch = -1
    for epoch in range(config.epochs):
        model.train()
        # Permutation is generated on CPU so the shuffle order is identical
        # regardless of device; only the arithmetic moves.
        permutation = torch.randperm(len(train_tensor), generator=generator).to(target)
        epoch_losses: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            batch = train_tensor[permutation[start:start + config.batch_size]]
            optimizer.zero_grad()
            total, parts = model.loss(batch)
            total.backward()
            optimizer.step()
            # Renormalise after the step so the dictionary stays unit-norm.
            model.normalize_decoder_()
            epoch_losses.append(parts["mse"])
        model.eval()
        with torch.no_grad():
            _, validation_parts = model.loss(validation_tensor)
        history.append({"epoch": epoch, "train_mse": float(np.mean(epoch_losses)),
                        "validation_mse": validation_parts["mse"],
                        "validation_l1": validation_parts["l1"]})
        if validation_parts["mse"] < best_validation:
            best_validation = validation_parts["mse"]
            best_epoch = epoch
            # Deep copy, so a later epoch cannot mutate the saved checkpoint.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise ResearchIntegrityError("SAE training produced no checkpoint")
    model.load_state_dict(best_state)
    model.freeze()
    return model, {
        "normalizer": normalizer, "history": history, "device": str(target),
        "selected_epoch": best_epoch, "selected_on": "validation reconstruction MSE",
        "best_validation_mse": best_validation,
        "labels_used_in_training": False,
        "expansion_factor": config.expansion_factor(x_train.shape[1]),
        "config": config.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sae_diagnostics(
    model: SparseAutoencoder, x: np.ndarray, normalizer: dict[str, Any] | None,
    *, partition: str = "unspecified",
) -> dict[str, Any]:
    """Reconstruction, variance explained, sparsity, and dead-feature reporting."""
    tensor = torch.from_numpy(apply_input_normalizer(x, normalizer)).to(_module_device(model))
    reconstruction, code, _ = model.forward(tensor)
    residual = (tensor - reconstruction).pow(2).sum().item()
    variance = (tensor - tensor.mean(dim=0, keepdim=True)).pow(2).sum().item()
    active = (code > 0)
    frequency = active.float().mean(dim=0).cpu().numpy()
    return {
        "partition": partition, "n_samples": int(len(tensor)),
        "n_features": int(model.n_features),
        "input_dim": int(model.input_dim),
        "expansion_factor": model.config.expansion_factor(model.input_dim),
        "reconstruction_mse": float(F.mse_loss(reconstruction, tensor).item()),
        "fraction_variance_unexplained": float(residual / variance) if variance > 0 else None,
        "fraction_variance_explained": float(1.0 - residual / variance) if variance > 0 else None,
        "l0_mean": float(active.float().sum(dim=-1).mean().item()),
        "l0_std": float(active.float().sum(dim=-1).std(unbiased=False).item()),
        "device": str(_module_device(model)),
        "dead_feature_fraction": float((frequency <= model.config.dead_feature_threshold).mean()),
        "n_dead_features": int((frequency <= model.config.dead_feature_threshold).sum()),
        "activation_frequency": {
            "mean": float(frequency.mean()), "median": float(np.median(frequency)),
            "max": float(frequency.max()), "min": float(frequency.min()),
        },
        "code_magnitude_mean": float(code[active].mean().item()) if active.any() else 0.0,
    }


@torch.no_grad()
def feature_activations(
    model: SparseAutoencoder, x: np.ndarray, normalizer: dict[str, Any] | None,
) -> np.ndarray:
    tensor = torch.from_numpy(apply_input_normalizer(x, normalizer)).to(_module_device(model))
    _, code = model.encode(tensor)
    return code.cpu().numpy()


# --------------------------------------------------------------------------- #
# Feature ranking and held-out evaluation
# --------------------------------------------------------------------------- #

def rank_features(
    model: SparseAutoencoder, x: np.ndarray, y: np.ndarray,
    normalizer: dict[str, Any] | None, *, min_activation_frequency: float = 0.01,
) -> pd.DataFrame:
    """Rank frozen features by label separability, on selection data only.

    Labels are used here and only here before the test evaluation. The SAE is
    already frozen, so this ranks a dictionary that was learned without labels --
    which is the distinction the legacy supervised autoencoder collapsed.
    """
    codes = feature_activations(model, x, normalizer)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ResearchIntegrityError("Feature ranking needs both label classes")
    rows: list[dict[str, Any]] = []
    for index in range(codes.shape[1]):
        column = codes[:, index]
        frequency = float((column > 0).mean())
        if frequency < min_activation_frequency:
            # Too rare to rank meaningfully; recorded rather than dropped.
            rows.append({"feature": index, "activation_frequency": frequency,
                         "auroc": None, "status": "too_rare",
                         "mean_active_positive": None, "mean_active_negative": None})
            continue
        rows.append({
            "feature": index, "activation_frequency": frequency, "status": "ranked",
            "auroc": float(roc_auc_score(y, column)),
            "mean_active_positive": float(column[y == 1].mean()),
            "mean_active_negative": float(column[y == 0].mean()),
        })
    table = pd.DataFrame(rows)
    # Separability is direction-agnostic: |AUROC - 0.5| is the ranking key.
    table["abs_auroc_gap"] = (table["auroc"] - 0.5).abs()
    return table.sort_values("abs_auroc_gap", ascending=False, na_position="last").reset_index(drop=True)


def evaluate_features_on_test(
    model: SparseAutoencoder, x_test: np.ndarray, y_test: np.ndarray,
    normalizer: dict[str, Any] | None, feature_indices: Sequence[int],
) -> dict[str, Any]:
    """Score already-selected features once on the frozen test partition."""
    if not len(feature_indices):
        raise ResearchIntegrityError("No features were selected for test evaluation")
    fingerprint_before = model.parameter_fingerprint()
    codes = feature_activations(model, x_test, normalizer)
    y_test = np.asarray(y_test, dtype=int)
    if len(np.unique(y_test)) != 2:
        raise ResearchIntegrityError("Test evaluation needs both label classes")
    per_feature = []
    for index in feature_indices:
        column = codes[:, int(index)]
        entry: dict[str, Any] = {"feature": int(index),
                                 "activation_frequency": float((column > 0).mean())}
        if len(np.unique(column)) < 2:
            entry.update({"status": "constant_on_test", "auroc": None})
        else:
            entry.update({"status": "ok", "auroc": float(roc_auc_score(y_test, column))})
        per_feature.append(entry)
    return {
        "selection_partition": "train+validation", "evaluation_partition": "test",
        "n_test": int(len(y_test)), "features": per_feature,
        "sae_unchanged_by_evaluation": model.parameter_fingerprint() == fingerprint_before,
    }


def top_activating_examples(
    model: SparseAutoencoder, x: np.ndarray, df: pd.DataFrame,
    normalizer: dict[str, Any] | None, feature_index: int, *, k: int = 10,
    metadata_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Top and contrast examples for one feature, with metadata, for inspection.

    Deliberately returns no semantic label. A feature that correlates with the
    bluff label is not thereby a "deception feature"; naming it is a human
    judgement made after reading these examples.
    """
    codes = feature_activations(model, x, normalizer)
    if not 0 <= feature_index < codes.shape[1]:
        raise ResearchIntegrityError(f"feature_index {feature_index} out of range")
    column = codes[:, feature_index]
    order = np.argsort(-column)
    columns = ["sample_id", "label"] + [c for c in metadata_columns if c in df.columns]

    def rows(indices: np.ndarray) -> list[dict[str, Any]]:
        return [{**{c: _jsonable(df.iloc[int(i)][c]) for c in columns},
                 "activation": float(column[int(i)]),
                 # Truncated: full prompts are large and may be private.
                 "statement_preview": str(df.iloc[int(i)]["statement"])[:240]}
                for i in indices]

    active = order[column[order] > 0]
    return {
        "feature": int(feature_index),
        "interpretation": "not assigned; requires human inspection",
        "activation_frequency": float((column > 0).mean()),
        "top_examples": rows(active[:k]),
        # Contrast set: lowest activations, which for a sparse code are zeros.
        "contrast_examples": rows(order[-k:][::-1]),
        "label_means": {
            "active": float(df.iloc[active]["label"].mean()) if len(active) else None,
            "inactive": float(df.iloc[order[column[order] <= 0]]["label"].mean())
                        if (column <= 0).any() else None,
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# Seed stability
# --------------------------------------------------------------------------- #

def seed_stability(
    x_train: np.ndarray, x_validation: np.ndarray, config: SAEConfig, seeds: Sequence[int],
    *, device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """How reproducible is the dictionary across training seeds?

    For each pair of runs, every feature in one is matched to its most similar
    feature in the other by absolute cosine similarity. A dictionary that is
    stable across seeds has a high mean max-similarity; a low value means the
    features are seed artifacts and should not be interpreted individually.
    """
    if len(seeds) < 2:
        raise ResearchIntegrityError("Seed stability needs at least two seeds")
    dictionaries = []
    for seed in seeds:
        model, _ = train_sae(x_train, x_validation,
                             SAEConfig(**{**config.to_dict(), "seed": int(seed)}), device=device)
        dictionaries.append(model.decoder_directions())
    pairs: list[dict[str, Any]] = []
    for i in range(len(dictionaries)):
        for j in range(i + 1, len(dictionaries)):
            similarity = np.abs(dictionaries[i] @ dictionaries[j].T)
            best = similarity.max(axis=1)
            pairs.append({"seeds": [int(seeds[i]), int(seeds[j])],
                          "mean_max_cosine": float(best.mean()),
                          "median_max_cosine": float(np.median(best)),
                          "fraction_above_0.9": float((best > 0.9).mean())})
    return {"seeds": [int(s) for s in seeds], "n_pairs": len(pairs), "pairs": pairs,
            "mean_max_cosine_across_pairs": float(np.mean([p["mean_max_cosine"] for p in pairs]))}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_sae_experiment(
    activations: np.ndarray, df: pd.DataFrame, manifest: dict[str, Any],
    config: PaperConfig, sae_config: SAEConfig, *, layer: int,
    top_n_features: int = 5, stability_seeds: Sequence[int] = (),
    metadata_columns: Iterable[str] = (),
    output_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Unsupervised train, freeze, rank on selection data, evaluate once on test."""
    index = {sid: i for i, sid in enumerate(df.sample_id)}
    def positions(part: str) -> np.ndarray:
        return np.array([index[s] for s in manifest[part]])

    train_pos, validation_pos, test_pos = (positions("train"), positions("validation"),
                                           positions("test"))
    if len(validation_pos) == 0:
        raise ResearchIntegrityError("SAE epoch selection requires a non-empty validation partition")
    x_train = activations[train_pos, layer]
    x_validation = activations[validation_pos, layer]
    x_test = activations[test_pos, layer]
    labels = df.label.to_numpy(int)

    model, training = train_sae(x_train, x_validation, sae_config, device=device)
    normalizer = training["normalizer"]
    fingerprint = model.parameter_fingerprint()

    diagnostics = {
        part: sae_diagnostics(model, data, normalizer, partition=part)
        for part, data in (("train", x_train), ("validation", x_validation), ("test", x_test))
    }

    # Ranking uses train + validation only; test is untouched until the next step.
    selection_pos = np.concatenate([train_pos, validation_pos])
    ranking = rank_features(model, activations[selection_pos, layer], labels[selection_pos], normalizer)
    ranked = ranking[ranking.status == "ranked"]
    selected = [int(f) for f in ranked.feature.head(top_n_features)]
    held_out = (evaluate_features_on_test(model, x_test, labels[test_pos], normalizer, selected)
                if selected else {"status": "not_run", "reason": "no feature met the frequency floor"})

    examples = [
        top_activating_examples(model, activations[selection_pos, layer],
                                df.iloc[selection_pos].reset_index(drop=True), normalizer,
                                feature, metadata_columns=metadata_columns)
        for feature in selected
    ]

    result: dict[str, Any] = {
        "layer": int(layer),
        "unsupervised": True,
        "labels_used_for": "feature ranking on train+validation only, after the SAE was frozen",
        "sae_config": sae_config.to_dict(),
        "expansion_factor": training["expansion_factor"],
        "training": {k: v for k, v in training.items() if k != "normalizer"},
        "input_normalizer": {k: v for k, v in (normalizer or {}).items() if k != "mean"},
        "diagnostics": diagnostics,
        "feature_ranking_top": ranked.head(max(top_n_features * 4, 20)).to_dict("records"),
        "n_features_too_rare_to_rank": int((ranking.status == "too_rare").sum()),
        "selected_features": selected,
        "held_out_feature_evaluation": held_out,
        "feature_examples": examples,
        "sae_frozen_before_evaluation": True,
        "sae_unchanged_after_evaluation": model.parameter_fingerprint() == fingerprint,
    }
    if stability_seeds:
        result["seed_stability"] = seed_stability(x_train, x_validation, sae_config,
                                                 stability_seeds, device=device)

    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        # CPU state dict: a CUDA-tensor checkpoint cannot be loaded on a CPU box.
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "input_dim": model.input_dim,
                    "sae_config": sae_config.to_dict(), "normalizer": normalizer,
                    "layer": int(layer), "unsupervised": True,
                    "trained_on_device": str(_module_device(model))}, root / "sae_model.pt")
        ranking.to_csv(root / "sae_feature_ranking.csv", index=False)
        (root / "sae_feature_examples.json").write_text(
            json.dumps(examples, indent=2) + "\n", encoding="utf-8")
        result["artifacts"] = {
            "model": str(root / "sae_model.pt"),
            "feature_ranking_csv": str(root / "sae_feature_ranking.csv"),
            "feature_examples_json": str(root / "sae_feature_examples.json"),
        }
    return result


__all__ = [
    "ACTIVATIONS", "SAEConfig", "SparseAutoencoder", "apply_input_normalizer",
    "resolve_device",
    "evaluate_features_on_test", "feature_activations", "fit_input_normalizer",
    "rank_features", "run_sae_experiment", "sae_diagnostics", "seed_stability",
    "top_activating_examples", "train_sae",
]
