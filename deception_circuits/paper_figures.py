"""Publication figures, generated only from real result artifacts.

Reviewer B found the previous figures unclear, so the conventions here are
deliberate rather than defaults:

- **No figure is ever produced from absent data.** Each builder returns
  ``status: "not_run"`` with a reason when its source artifact is missing. There
  is no mock or sample-data path, so a publication directory cannot fill up with
  plausible-looking figures that correspond to nothing.
- **Every figure ships its source data** as a CSV next to the image, and
  ``figures_manifest.json`` records the artifact each figure came from together
  with that artifact's SHA-256. A reader can therefore trace any plotted number
  back to the run that produced it.
- **Metadata travels with the figure.** Model, dataset, split, sample size, layer
  definition, and uncertainty method go in the manifest caption; only a short
  subtitle is drawn on the image, because the brief rules out burying unexplained
  captions inside a raster.
- **One axis, never two.** Measures on different scales get separate panels.
- **Identity is never colour alone.** Every series carries a distinct marker and
  a legend entry; the categorical hues are used in fixed order and never cycled.

The categorical palette below was checked with the data-viz validator on a light
surface: all checks pass, worst adjacent CVD separation dE 9.1 (protan) / 5.8
(tritan). A tritan value in the floor band is only admissible with secondary
encoding, which is why per-series markers are mandatory here rather than
cosmetic, and why the accompanying CSV acts as the table view that discharges the
sub-3:1 contrast warning on three of the slots.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: figures are files, never windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paper import ResearchIntegrityError

# Fixed categorical order. Never cycled: a 7th series folds into "other" or the
# figure becomes small multiples.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "^", "D", "v", "P")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

ARTIFACTS = {
    "probe": "probe_results.json",
    "confound": "confound_results.json",
    "transfer": "cross_context_results.json",
    "causal": "causal_results.json",
    "sae": "sae_results.json",
    "split": "split_manifest.json",
    "metadata": "run_metadata.json",
}

@dataclass(frozen=True)
class FigureContext:
    """Provenance stamped onto every figure and its manifest entry."""

    model: str
    dataset: str
    dataset_sha256: str | None
    git_commit: str | None
    experiment_name: str

    def subtitle(self, *, split: str, n: int | None, layer: int | None = None) -> str:
        parts = [self.model, f"{split} split"]
        if n is not None:
            parts.append(f"n={n}")
        if layer is not None:
            parts.append(f"layer {layer}")
        return "  |  ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Load whichever result artifacts exist, recording each one's checksum."""
    root = Path(output_dir)
    loaded: dict[str, Any] = {"root": str(root), "present": {}, "checksums": {}}
    for key, filename in ARTIFACTS.items():
        path = root / filename
        if path.is_file():
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
            loaded["present"][key] = str(path)
            loaded["checksums"][key] = _sha256(path)
        else:
            loaded[key] = None
    return loaded


def build_figure_context(artifacts: dict[str, Any]) -> FigureContext:
    metadata = artifacts.get("metadata") or {}
    config = metadata.get("config") or {}
    split = artifacts.get("split") or {}
    return FigureContext(
        model=str(config.get("subject_model") or "unspecified model"),
        dataset=str(config.get("dataset_path") or "unspecified dataset"),
        dataset_sha256=split.get("dataset_sha256"),
        git_commit=metadata.get("git_commit"),
        experiment_name=str(config.get("experiment_name") or "unnamed"),
    )


def _pretty(name: str) -> str:
    """Humanise an identifier for an axis label: readers are not reading code."""
    return str(name).replace("_", " ")


def _style_axes(ax: plt.Axes, *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)  # recessive grid, behind the marks
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def _finish(fig: plt.Figure, ax: plt.Axes | Sequence[plt.Axes], title: str, subtitle: str) -> None:
    """Title, provenance strip, and surface, laid out without a dead band.

    The provenance line rides on the axes title for single-panel figures so that
    ``tight_layout`` accounts for its height; multi-panel figures already use
    their axes titles for panel names, so it goes into the suptitle instead.
    Either way it stays a short strip -- the full caption lives in the manifest,
    because the brief rules out burying captions in a raster.
    """
    fig.patch.set_facecolor(SURFACE)
    axes = list(ax) if isinstance(ax, (list, tuple, np.ndarray)) else [ax]
    for axis in axes:
        axis.set_facecolor(SURFACE)
    if len(axes) == 1:
        # The title sits directly above the provenance strip: the strip rides on
        # the axes title, so tight_layout already reserves its height and the
        # suptitle only needs the remaining sliver.
        axes[0].set_title(subtitle, fontsize=8.5, color=INK_SECONDARY, pad=8)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.suptitle(title, fontsize=12.5, color=INK_PRIMARY, y=0.965)
    else:
        fig.suptitle(f"{title}\n{subtitle}", fontsize=11, color=INK_PRIMARY, y=0.99,
                     linespacing=1.6)
        fig.tight_layout(rect=(0, 0, 1, 0.90))


def _save(fig: plt.Figure, data: pd.DataFrame, name: str, figures_dir: Path,
          *, caption: str, source: str | None, checksum: str | None,
          uncertainty: str) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    image = figures_dir / f"{name}.png"
    table = figures_dir / f"{name}.csv"
    fig.savefig(image, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    data.to_csv(table, index=False)
    return {"status": "ok", "figure": str(image), "source_data": str(table),
            "caption": caption, "generated_from": source,
            "source_sha256": checksum, "uncertainty_method": uncertainty}


def _skip(name: str, reason: str) -> dict[str, Any]:
    return {"status": "not_run", "figure": None, "source_data": None, "reason": reason,
            "name": name}


# --------------------------------------------------------------------------- #
# Probe figures
# --------------------------------------------------------------------------- #

def figure_layerwise_probe(
    artifacts: dict[str, Any],
    context: FigureContext,
    figures_dir: Path,
) -> dict[str, Any]:
    """Validation AUROC by physical transformer layer across seeds."""
    probe = artifacts.get("probe")
    if not probe:
        return _skip("layerwise_probe", "probe_results.json is absent")

    runs = probe.get("runs") or []
    curves = [r.get("validation_auroc_by_layer") for r in runs]
    if not curves or any(c is None for c in curves):
        return _skip(
            "layerwise_probe",
            "probe runs do not record validation_auroc_by_layer",
        )

    matrix = np.asarray(curves, dtype=float)

    raw_layer_indices = probe.get("activation_layer_indices")
    if raw_layer_indices is None:
        return _skip(
            "layerwise_probe",
            "probe_results.json does not record activation_layer_indices, so "
            "tensor axes cannot be mapped safely to physical transformer layers",
        )

    physical_layers = np.asarray(
        [int(layer) for layer in raw_layer_indices],
        dtype=int,
    )

    if len(physical_layers) != matrix.shape[1]:
        return _skip(
            "layerwise_probe",
            "activation_layer_indices length does not match the number of "
            "layerwise validation scores",
        )

    if len(set(physical_layers.tolist())) != len(physical_layers):
        return _skip(
            "layerwise_probe",
            "activation_layer_indices contains duplicate physical layers",
        )

    mean = matrix.mean(axis=0)
    spread = (
        matrix.std(axis=0, ddof=1)
        if len(matrix) > 1
        else np.zeros(matrix.shape[1])
    )

    selected_physical_layers: list[int] = []
    for run in runs:
        if "selected_physical_layer" in run:
            selected_physical_layers.append(
                int(run["selected_physical_layer"])
            )
            continue

        selected_axis = run.get(
            "selected_layer_index",
            run.get("selected_layer"),
        )
        if selected_axis is None:
            return _skip(
                "layerwise_probe",
                "a probe run does not record its selected layer",
            )

        selected_axis = int(selected_axis)
        if not 0 <= selected_axis < len(physical_layers):
            return _skip(
                "layerwise_probe",
                "a probe run selected an activation tensor axis outside the "
                "recorded activation_layer_indices mapping",
            )

        selected_physical_layers.append(
            int(physical_layers[selected_axis])
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        physical_layers,
        mean,
        color=PALETTE[0],
        marker=MARKERS[0],
        markersize=5,
        linewidth=2,
        label=f"mean validation AUROC ({len(runs)} seeds)",
    )
    ax.fill_between(
        physical_layers,
        mean - spread,
        mean + spread,
        color=PALETTE[0],
        alpha=0.18,
        label="+/-1 SD across seeds",
    )

    modal_physical_layer = max(
        set(selected_physical_layers),
        key=selected_physical_layers.count,
    )
    ax.axvline(
        modal_physical_layer,
        color=PALETTE[1],
        linestyle="--",
        linewidth=1.5,
        label=(
            f"selected physical layer {modal_physical_layer} "
            "(modal across seeds)"
        ),
    )
    ax.axhline(
        0.5,
        color=INK_SECONDARY,
        linewidth=1,
        linestyle=":",
        label="chance (AUROC 0.5)",
    )
    ax.set_ylim(0.0, 1.02)
    _style_axes(
        ax,
        xlabel="physical transformer layer",
        ylabel="validation AUROC",
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    _finish(
        fig,
        ax,
        "Layerwise probe performance (validation)",
        context.subtitle(
            split="validation",
            n=probe.get("n_validation"),
            layer=modal_physical_layer,
        ),
    )

    data = pd.DataFrame({
        "activation_layer_index": np.arange(matrix.shape[1]),
        "physical_layer": physical_layers,
        "mean_validation_auroc": mean,
        "sd_across_seeds": spread,
    })

    for i, run in enumerate(runs):
        data[f"seed_{run['seed']}_validation_auroc"] = matrix[i]

    return _save(
        fig,
        data,
        "layerwise_probe",
        figures_dir,
        caption=(
            f"Validation AUROC by physical transformer layer for "
            f"{context.model}, {len(runs)} seeds. Layer selection used the "
            f"validation partition only; the frozen selection was evaluated "
            f"once on test. The modal selected physical transformer layer is "
            f"{modal_physical_layer}. Shaded band is +/-1 SD across seeds. "
            f"Dataset {context.dataset}."
        ),
        source=artifacts["present"].get("probe"),
        checksum=artifacts["checksums"].get("probe"),
        uncertainty="standard deviation across probe-fitting seeds",
    )

def figure_learning_curve(
    artifacts: dict[str, Any],
    context: FigureContext,
    figures_dir: Path,
) -> dict[str, Any]:
    """Test AUROC against training-set size in groups, over independent subsamples."""

    probe = artifacts.get("probe")
    curve = (probe or {}).get("learning_curve")
    if not curve:
        return _skip("learning_curve", "probe_results.json has no learning_curve "
                                       "(set learning_curve_sizes in the config)")
    points = [p for p in curve.get("points", []) if p.get("status") == "ok"]
    if not points:
        return _skip("learning_curve", "no learning-curve size completed; see the point reasons")
    sizes = [p["n_train_groups"] for p in points]
    means = [p["test_auroc_mean"] for p in points]
    errors = [p["test_auroc_std"] for p in points]
    replicates = points[0].get("n_replicates")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(sizes, means, yerr=errors, color=PALETTE[0], marker=MARKERS[0], markersize=6,
                linewidth=2, capsize=4,
                label=f"mean test AUROC ({replicates} independent subsamples)")
    ax.axhline(0.5, color=INK_SECONDARY, linewidth=1, linestyle=":", label="chance (AUROC 0.5)")
    ax.set_ylim(0.0, 1.02)
    _style_axes(ax, xlabel="training-set size (independent groups / poker hands)",
                ylabel="test AUROC")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _finish(fig, ax, "Learning curve over independent group-level subsamples",
            context.subtitle(split="test", n=probe.get("n_test")))

    data = pd.DataFrame({"n_train_groups": sizes, "test_auroc_mean": means,
                         "test_auroc_std": errors,
                         "n_replicates": [p["n_replicates"] for p in points],
                         "test_auroc_min": [p["test_auroc_range"][0] for p in points],
                         "test_auroc_max": [p["test_auroc_range"][1] for p in points]})
    return _save(fig, data, "learning_curve", figures_dir,
                 caption=(f"Test AUROC as a function of training-set size for {context.model}. "
                          f"Subsampling is at group (hand) level with {replicates} independent "
                          f"draws per size, so a smaller training set is fewer hands rather than "
                          f"fewer rows from the same hands. Error bars are +/-1 SD across draws."),
                 source=artifacts["present"].get("probe"),
                 checksum=artifacts["checksums"].get("probe"),
                 uncertainty="standard deviation across independent subsamples")


def figure_nuisance_baselines(artifacts: dict[str, Any], context: FigureContext,
                              figures_dir: Path) -> dict[str, Any]:
    """The probe against every trivial and nuisance baseline on the same test split."""
    probe = artifacts.get("probe")
    if not probe:
        return _skip("nuisance_baselines", "probe_results.json is absent")
    baselines = probe.get("baselines") or {}
    rows: list[dict[str, Any]] = [
        {"comparison": "hidden-state probe", "auroc": probe.get("test_auroc_mean"),
         "kind": "probe"},
    ]
    if isinstance(baselines.get("prompt_text"), dict):
        rows.append({"comparison": "prompt text only", "kind": "baseline",
                     "auroc": baselines["prompt_text"].get("auroc")})
    if isinstance(baselines.get("nuisance_only"), dict) and "auroc" in baselines["nuisance_only"]:
        rows.append({"comparison": "nuisance metadata only", "kind": "baseline",
                     "auroc": baselines["nuisance_only"]["auroc"]})
    if isinstance(baselines.get("response_text_diagnostic"), dict):
        rows.append({"comparison": "response text (diagnostic, not a claim)",
                     "kind": "diagnostic",
                     "auroc": baselines["response_text_diagnostic"].get("auroc")})
    data = pd.DataFrame([r for r in rows if r["auroc"] is not None])
    if len(data) < 2:
        return _skip("nuisance_baselines", "fewer than two comparable baselines were recorded")

    colors = {"probe": PALETTE[0], "baseline": PALETTE[1], "diagnostic": PALETTE[4]}
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(data) + 2.4))
    positions = np.arange(len(data))
    ax.barh(positions, data["auroc"], color=[colors[k] for k in data["kind"]], height=0.62)
    ax.set_yticks(positions)
    ax.set_yticklabels(data["comparison"], fontsize=9, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.axvline(0.5, color=INK_SECONDARY, linewidth=1, linestyle=":", label="chance (AUROC 0.5)")
    for position, value in zip(positions, data["auroc"]):
        # Direct labels: magnitude is never colour-alone.
        ax.text(value + 0.012, position, f"{value:.3f}", va="center", fontsize=9,
                color=INK_PRIMARY)
    ax.set_xlim(0.0, 1.12)
    _style_axes(ax, xlabel="test AUROC", ylabel="")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower right")
    _finish(fig, ax, "Probe versus trivial and nuisance baselines",
            context.subtitle(split="test", n=probe.get("n_test")))
    return _save(fig, data, "nuisance_baselines", figures_dir,
                 caption=("Test AUROC for the hidden-state probe and every baseline fitted on the "
                          "same training partition. The response-text bar is a leakage diagnostic, "
                          "not a result: on a dataset whose response column encodes the label it "
                          "will sit near 1.0 by construction. A probe that does not clearly exceed "
                          "the nuisance-metadata bar is not evidence of a deception representation."),
                 source=artifacts["present"].get("probe"),
                 checksum=artifacts["checksums"].get("probe"),
                 uncertainty="point estimates; see the layerwise figure for seed spread")


# --------------------------------------------------------------------------- #
# Confound figures
# --------------------------------------------------------------------------- #

def figure_confound_subsets(artifacts: dict[str, Any], context: FigureContext,
                            figures_dir: Path) -> dict[str, Any]:
    """Probe AUROC on each matched subset, with the controls that could not run named."""
    confound = artifacts.get("confound")
    if not confound:
        return _skip("confound_subsets", "confound_results.json is absent (run analyze-confounds)")
    rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for name, entry in (confound.get("subsets") or {}).items():
        if entry.get("status") == "not_run":
            unavailable.append(f"{name}: {entry.get('reason', 'unspecified')}")
            continue
        probe_entry = entry.get("probe") or {}
        if probe_entry.get("status") != "ok":
            unavailable.append(f"{name}: {probe_entry.get('reason', 'probe did not run')}")
            continue
        interval = probe_entry.get("test_auroc_grouped_ci") or {}
        rows.append({"subset": name, "auroc": probe_entry["auroc"],
                     "ci_lower": interval.get("lower"), "ci_upper": interval.get("upper"),
                     "n_test": probe_entry.get("n_test")})
    if not rows:
        return _skip("confound_subsets",
                     "no matched subset produced a probe result. " + "; ".join(unavailable))

    data = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    unadjusted = ((confound.get("controlled_probe") or {}).get("unadjusted") or {}).get("auroc")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 2.6))
    positions = np.arange(len(data))
    lower = data["auroc"] - data["ci_lower"].fillna(data["auroc"])
    upper = data["ci_upper"].fillna(data["auroc"]) - data["auroc"]
    ax.errorbar(data["auroc"], positions, xerr=[lower, upper], fmt=MARKERS[0],
                color=PALETTE[0], markersize=7, capsize=4, linewidth=0,
                elinewidth=1.6, label="matched subset (grouped bootstrap 95% CI)")
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{_pretty(s)}  (n={int(n)})" for s, n in zip(data["subset"], data["n_test"])],
                       fontsize=9, color=INK_PRIMARY)
    ax.invert_yaxis()
    if unadjusted is not None:
        # Explicitly named, so it is not an unexplained reference curve.
        ax.axvline(unadjusted, color=PALETTE[1], linestyle="--", linewidth=1.5,
                   label=f"full unmatched test set ({unadjusted:.3f})")
    ax.axvline(0.5, color=INK_SECONDARY, linewidth=1, linestyle=":", label="chance (AUROC 0.5)")
    ax.set_xlim(0.0, 1.05)
    _style_axes(ax, xlabel="test AUROC within matched subset", ylabel="")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")
    _finish(
        fig,
        ax,
        "Probe performance under poker confound controls",
        context.subtitle(
            split="test",
            n=None,
            layer=confound.get("selected_physical_layer"),
        ),
    )

    caption = (
        "Probe AUROC inside each nuisance-matched subset, at the physical transformer "
        "layer selected on validation. Matching equalises label counts within every "
        "stratum, so a drop relative to the unmatched set is the portion of apparent "
        "performance that the nuisance variable explained."
    )
    if unavailable:
        caption += " Controls that could not run: " + "; ".join(unavailable)
    return _save(fig, data, "confound_subsets", figures_dir, caption=caption,
                 source=artifacts["present"].get("confound"),
                 checksum=artifacts["checksums"].get("confound"),
                 uncertainty="grouped bootstrap 95% percentile CI")


def figure_cross_scenario(
    artifacts: dict[str, Any],
    context: FigureContext,
    figures_dir: Path,
) -> dict[str, Any]:
    """Cross-context transfer AUROC with source-only layer selection."""
    transfer = artifacts.get("transfer")

    if not transfer:
        return _skip(
            "cross_scenario",
            "cross_context_results.json is absent (run run-transfer)",
        )

    if transfer.get("status") != "ok":
        return _skip(
            "cross_scenario",
            transfer.get(
                "reason",
                "cross-context transfer did not complete",
            ),
        )

    cells = [
        cell
        for cell in transfer.get("cells", [])
        if cell.get("status") == "ok"
    ]

    if not cells:
        return _skip(
            "cross_scenario",
            "cross-context transfer produced no completed source-target cells",
        )

    contexts = [str(value) for value in transfer.get("contexts", [])]
    if len(contexts) < 2:
        return _skip(
            "cross_scenario",
            "cross-context transfer requires at least two contexts",
        )

    context_to_index = {
        name: index
        for index, name in enumerate(contexts)
    }

    matrix = np.full(
        (len(contexts), len(contexts)),
        np.nan,
        dtype=float,
    )

    rows: list[dict[str, Any]] = []

    for cell in cells:
        source = str(cell["source_context"])
        target = str(cell["target_context"])

        if source not in context_to_index or target not in context_to_index:
            return _skip(
                "cross_scenario",
                "a completed transfer cell references a context absent from "
                "the artifact context list",
            )

        target_metrics = cell.get("target_test") or {}
        auroc = target_metrics.get("auroc")

        if auroc is None:
            return _skip(
                "cross_scenario",
                "a completed transfer cell is missing target-test AUROC",
            )

        source_index = context_to_index[source]
        target_index = context_to_index[target]
        matrix[source_index, target_index] = float(auroc)

        interval = cell.get("target_test_auroc_grouped_ci") or {}

        rows.append({
            "source_context": source,
            "target_context": target,
            "selected_layer_index": cell.get("selected_layer_index"),
            "selected_physical_layer": cell.get(
                "selected_physical_layer"
            ),
            "source_validation_auroc": cell.get(
                "source_validation_auroc"
            ),
            "target_test_auroc": float(auroc),
            "target_test_ci_lower": interval.get("lower"),
            "target_test_ci_upper": interval.get("upper"),
            "n_source_train": cell.get("n_source_train"),
            "n_source_validation": cell.get("n_source_validation"),
            "n_target_test": cell.get("n_target_test"),
        })

    data = pd.DataFrame(rows).sort_values(
        ["source_context", "target_context"]
    ).reset_index(drop=True)

    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(
        figsize=(
            max(6.0, 1.2 * len(contexts) + 2.5),
            max(5.0, 1.0 * len(contexts) + 2.5),
        )
    )

    image = ax.imshow(
        masked,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(contexts)))
    ax.set_yticks(np.arange(len(contexts)))
    ax.set_xticklabels(contexts, rotation=30, ha="right")
    ax.set_yticklabels(contexts)

    for source_index in range(len(contexts)):
        for target_index in range(len(contexts)):
            value = matrix[source_index, target_index]

            if np.isfinite(value):
                ax.text(
                    target_index,
                    source_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            elif source_index == target_index:
                ax.text(
                    target_index,
                    source_index,
                    "within\ncontext",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK_SECONDARY,
                )

    ax.set_xlabel(
        "held-out target context",
        color=INK_SECONDARY,
    )
    ax.set_ylabel(
        "source context used for training and validation",
        color=INK_SECONDARY,
    )

    ax.tick_params(
        colors=INK_SECONDARY,
        labelsize=9,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "target-test AUROC",
        color=INK_SECONDARY,
    )
    colorbar.ax.tick_params(
        colors=INK_SECONDARY,
        labelsize=8,
    )

    _finish(
        fig,
        ax,
        "Cross-context probe generalization",
        context.subtitle(
            split="held-out target test",
            n=None,
        ),
    )

    return _save(
        fig,
        data,
        "cross_scenario",
        figures_dir,
        caption=(
            "Cross-context test AUROC after fitting probes on the source "
            "context training partition and selecting the activation layer "
            "using source-context validation only. Each off-diagonal cell "
            "evaluates the frozen source probe on a different context from "
            "the held-out global test partition. Target-test labels are never "
            "used for model or layer selection."
        ),
        source=artifacts["present"].get("transfer"),
        checksum=artifacts["checksums"].get("transfer"),
        uncertainty=(
            "grouped bootstrap 95% percentile CI recorded in the source CSV"
        ),
    )

# --------------------------------------------------------------------------- #
# Causal figures
# --------------------------------------------------------------------------- #

def _causal_effects(causal: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, entry in (causal.get("effects") or {}).items():
        for strength, stats in (entry.get("by_strength") or {}).items():
            rows.append({"condition": condition, "strength": float(strength),
                         "mean_difference": stats.get("mean_difference"),
                         "ci_lower": stats.get("ci_lower"), "ci_upper": stats.get("ci_upper"),
                         "n_pairs": stats.get("n_pairs"), "n_groups": stats.get("n_groups"),
                         "effect_size_dz": stats.get("effect_size_dz"),
                         "signed_consistency_rate": stats.get("signed_consistency_rate"),
                         "mean_relative_magnitude": stats.get("mean_relative_magnitude")})
    return pd.DataFrame(rows)


def figure_dose_response(artifacts: dict[str, Any], context: FigureContext,
                         figures_dir: Path) -> dict[str, Any]:
    """Paired endpoint change against steering strength, for the steering conditions."""
    causal = artifacts.get("causal")
    if not causal:
        return _skip("dose_response", "causal_results.json is absent (run run-interventions)")
    data = _causal_effects(causal)
    steering = data[data.condition.isin(["positive_steering", "negative_steering"])]
    if steering.empty or steering.strength.nunique() < 2:
        return _skip("dose_response",
                     "fewer than two strengths were evaluated for the steering conditions")

    fig, ax = plt.subplots(figsize=(7, 4))
    for index, (condition, block) in enumerate(steering.groupby("condition")):
        block = block.sort_values("strength")
        ax.errorbar(block.strength, block.mean_difference,
                    yerr=[block.mean_difference - block.ci_lower,
                          block.ci_upper - block.mean_difference],
                    color=PALETTE[index], marker=MARKERS[index], markersize=6, linewidth=2,
                    capsize=4, label=condition.replace("_", " "))
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=1, linestyle=":", label="no change")
    _style_axes(ax, xlabel="steering strength (multiples of the unit-norm direction)",
                ylabel=f"paired change in {_pretty(causal.get('primary_endpoint', 'endpoint'))}")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _finish(fig, ax, "Behavioral dose-response under steering",
            context.subtitle(split="held-out test", n=causal.get("n_prompts"),
                             layer=causal.get("layer")))
    return _save(fig, steering, "dose_response", figures_dir,
                 caption=(f"Paired change in {causal.get('primary_endpoint')} against steering "
                          f"strength on {causal.get('n_prompts')} held-out prompts "
                          f"({causal.get('n_groups')} groups) at layer {causal.get('layer')}. "
                          f"The endpoint is the model's own output distribution, not the probe "
                          f"score. Decoding: {causal.get('decoding')}. Error bars are paired "
                          f"grouped bootstrap 95% CIs."),
                 source=artifacts["present"].get("causal"),
                 checksum=artifacts["checksums"].get("causal"),
                 uncertainty="paired grouped bootstrap 95% percentile CI")


def figure_intervention_controls(artifacts: dict[str, Any], context: FigureContext,
                                 figures_dir: Path) -> dict[str, Any]:
    """Every condition side by side at its strongest tested dose."""
    causal = artifacts.get("causal")
    if not causal:
        return _skip("intervention_controls", "causal_results.json is absent")
    data = _causal_effects(causal)
    if data.empty:
        return _skip("intervention_controls", "causal_results.json records no effects")
    strongest = data.sort_values("strength").groupby("condition", as_index=False).last()
    strongest = strongest.sort_values("mean_difference", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(strongest) + 2.4))
    positions = np.arange(len(strongest))
    ax.errorbar(strongest.mean_difference, positions,
                xerr=[strongest.mean_difference - strongest.ci_lower,
                      strongest.ci_upper - strongest.mean_difference],
                fmt=MARKERS[0], color=PALETTE[0], markersize=7, capsize=4, linewidth=0,
                elinewidth=1.6, label="paired effect (grouped bootstrap 95% CI)")
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{_pretty(c)}  (strength {s:g})"
                        for c, s in zip(strongest.condition, strongest.strength)],
                       fontsize=9, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.axvline(0.0, color=INK_SECONDARY, linewidth=1, linestyle=":", label="no change")
    _style_axes(ax, xlabel=f"paired change in {_pretty(causal.get('primary_endpoint', 'endpoint'))}",
                ylabel="")
    # Interior upper-left: the largest positive effect sorts to the top row and
    # sits far right, so that corner is free. The band above the axes is already
    # taken by the provenance strip.
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    _finish(fig, ax, "Steering effect against every control condition",
            context.subtitle(split="held-out test", n=causal.get("n_prompts"),
                             layer=causal.get("layer")))
    return _save(fig, strongest, "intervention_controls", figures_dir,
                 caption=("Paired endpoint change for every condition at its strongest tested "
                          "dose. Random matched-norm, orthogonal, shuffled-label, wrong-layer, "
                          "wrong-token-position, and mismatched-prompt patching are controls: a "
                          "steering effect is only interpretable if it clearly exceeds them."),
                 source=artifacts["present"].get("causal"),
                 checksum=artifacts["checksums"].get("causal"),
                 uncertainty="paired grouped bootstrap 95% percentile CI")


def figure_intervention_quality(artifacts: dict[str, Any], context: FigureContext,
                                figures_dir: Path) -> dict[str, Any]:
    """Format adherence per condition, so degradation cannot pass as an effect."""
    causal = artifacts.get("causal")
    quality = (causal or {}).get("quality_controls")
    if not quality:
        return _skip("intervention_quality", "causal_results.json has no quality_controls")
    rows = [{"condition": condition, "strength": float(strength),
             "mean_valid_choice_mass": stats.get("mean_valid_choice_mass"),
             "min_valid_choice_mass": stats.get("min_valid_choice_mass"),
             "chose_positive_rate": stats.get("chose_positive_rate"), "n": stats.get("n")}
            for condition, by_strength in quality.items()
            for strength, stats in by_strength.items()]
    data = pd.DataFrame(rows)
    if data.empty:
        return _skip("intervention_quality", "quality_controls is empty")
    strongest = data.sort_values("strength").groupby("condition", as_index=False).last()
    strongest = strongest.sort_values("condition").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(strongest) + 3.0))
    positions = np.arange(len(strongest))
    labels = [c.replace("_", " ") for c in strongest.condition]
    # Two panels rather than one dual-axis chart: different scales, separate axes.
    axes[0].barh(positions, strongest.mean_valid_choice_mass, color=PALETTE[0], height=0.62)
    axes[0].set_title("answer-format adherence", fontsize=10, color=INK_PRIMARY)
    _style_axes(axes[0], xlabel="mean probability mass on the allowed options", ylabel="")
    axes[1].barh(positions, strongest.chose_positive_rate, color=PALETTE[2], height=0.62)
    axes[1].set_title("rate of choosing the positive option", fontsize=10, color=INK_PRIMARY)
    _style_axes(axes[1], xlabel="fraction of held-out prompts", ylabel="")
    for axis in axes:
        axis.set_yticks(positions)
        axis.invert_yaxis()
    axes[0].set_yticklabels(labels, fontsize=9, color=INK_PRIMARY)
    axes[1].set_yticklabels([])
    _finish(fig, list(axes), "Generation-quality controls beside the causal effects",
            context.subtitle(split="held-out test", n=causal.get("n_prompts")))
    return _save(fig, strongest, "intervention_quality", figures_dir,
                 caption=("Per-condition format adherence and positive-choice rate at the "
                          "strongest tested dose. An intervention that moves the endpoint while "
                          "collapsing format adherence has degraded the model rather than changed "
                          "the target behaviour."),
                 source=artifacts["present"].get("causal"),
                 checksum=artifacts["checksums"].get("causal"),
                 uncertainty="point estimates over the held-out prompts")


# --------------------------------------------------------------------------- #
# SAE figures
# --------------------------------------------------------------------------- #

def figure_sae_diagnostics(artifacts: dict[str, Any], context: FigureContext,
                           figures_dir: Path) -> dict[str, Any]:
    """Reconstruction, sparsity, and dead features per partition."""
    sae = artifacts.get("sae")
    if not sae:
        return _skip("sae_diagnostics", "sae_results.json is absent (run train-sae)")
    diagnostics = sae.get("diagnostics") or {}
    order = [p for p in ("train", "validation", "test") if p in diagnostics]
    if not order:
        return _skip("sae_diagnostics", "sae_results.json records no per-partition diagnostics")
    data = pd.DataFrame([{"partition": p, **{k: v for k, v in diagnostics[p].items()
                                             if not isinstance(v, dict)}} for p in order])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    # Variance explained can legitimately go negative (the SAE reconstructing worse
    # than the partition mean), so its axis is never clamped to zero: hiding that
    # would turn a failed reconstruction into a blank panel.
    panels = (("fraction_variance_explained", "fraction of variance explained", PALETTE[0], 1.05),
              ("l0_mean", "mean active features per example (L0)", PALETTE[2], None),
              ("dead_feature_fraction", "dead feature fraction", PALETTE[1], 1.05))
    positions = np.arange(len(order))
    for axis, (column, label, color, ceiling) in zip(axes, panels):
        values = data[column].astype(float)
        axis.bar(positions, values, color=color, width=0.6)
        lowest = float(min(0.0, values.min()))
        highest = float(ceiling) if ceiling is not None else float(values.max()) * 1.18
        span = max(highest - lowest, 1e-6)
        for position, value in zip(positions, values):
            # Labels sit outside the bar on whichever side the bar grows.
            axis.text(position, value + (0.02 if value >= 0 else -0.02) * span,
                      f"{value:.3g}", ha="center",
                      va="bottom" if value >= 0 else "top",
                      fontsize=9, color=INK_PRIMARY)
        axis.set_xticks(positions)
        axis.set_xticklabels(order, fontsize=9, color=INK_PRIMARY)
        axis.set_ylim(lowest - 0.09 * span, highest)
        if lowest < 0:
            axis.axhline(0.0, color=INK_SECONDARY, linewidth=1)
        _style_axes(axis, xlabel="", ylabel=label)
    _finish(fig, list(axes), "Frozen SAE diagnostics by partition",
            context.subtitle(split="all partitions", n=diagnostics[order[-1]].get("n_samples"),
                             layer=sae.get("layer")))
    return _save(fig, data, "sae_diagnostics", figures_dir,
                 caption=(f"Diagnostics for the unsupervised SAE at layer {sae.get('layer')}, "
                          f"{sae['sae_config']['n_features']} features "
                          f"(expansion {sae.get('expansion_factor'):.2f}x), "
                          f"{sae['sae_config']['activation']} activation. The SAE was trained "
                          f"without labels and frozen before these partitions were scored."),
                 source=artifacts["present"].get("sae"),
                 checksum=artifacts["checksums"].get("sae"),
                 uncertainty="none: single frozen dictionary, deterministic evaluation")


def figure_sae_features(artifacts: dict[str, Any], context: FigureContext,
                        figures_dir: Path) -> dict[str, Any]:
    """Selected features: selection-set separability against held-out separability."""
    sae = artifacts.get("sae")
    if not sae:
        return _skip("sae_features", "sae_results.json is absent")
    held_out = sae.get("held_out_feature_evaluation") or {}
    if held_out.get("evaluation_partition") != "test":
        return _skip("sae_features", "no held-out feature evaluation is recorded")
    ranking = {int(r["feature"]): r for r in sae.get("feature_ranking_top") or []}
    rows = []
    for entry in held_out.get("features") or []:
        feature = int(entry["feature"])
        rows.append({"feature": feature,
                     "selection_auroc": (ranking.get(feature) or {}).get("auroc"),
                     "test_auroc": entry.get("auroc"),
                     "test_activation_frequency": entry.get("activation_frequency"),
                     "status": entry.get("status")})
    data = pd.DataFrame(rows).dropna(subset=["selection_auroc", "test_auroc"])
    if data.empty:
        return _skip("sae_features", "no selected feature has both selection and test AUROC")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle=":",
            label="equal selection and test separability")
    ax.axhline(0.5, color=GRID, linewidth=1)
    ax.axvline(0.5, color=GRID, linewidth=1)
    ax.scatter(data.selection_auroc, data.test_auroc, s=90, color=PALETTE[0],
               marker=MARKERS[0], zorder=3, label="selected SAE feature")
    for _, row in data.iterrows():
        ax.annotate(f"#{int(row.feature)}", (row.selection_auroc, row.test_auroc),
                    textcoords="offset points", xytext=(7, 4), fontsize=8, color=INK_PRIMARY)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _style_axes(ax, xlabel="AUROC on train+validation (used for ranking)",
                ylabel="AUROC on test (frozen evaluation)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    _finish(fig, ax, "SAE features: selection versus held-out separability",
            context.subtitle(split="test", n=held_out.get("n_test"), layer=sae.get("layer")))
    return _save(fig, data, "sae_features", figures_dir,
                 caption=("Each point is one SAE feature selected by label separability on "
                          "train+validation and then scored once on test. Features far below the "
                          "diagonal were selection artifacts. No feature is assigned a semantic "
                          "name here: correlation with the bluff label does not make a feature a "
                          "deception feature."),
                 source=artifacts["present"].get("sae"),
                 checksum=artifacts["checksums"].get("sae"),
                 uncertainty="none: point estimates on the frozen test partition")


FIGURE_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "layerwise_probe": figure_layerwise_probe,
    "learning_curve": figure_learning_curve,
    "nuisance_baselines": figure_nuisance_baselines,
    "confound_subsets": figure_confound_subsets,
    "cross_scenario": figure_cross_scenario,
    "dose_response": figure_dose_response,
    "intervention_controls": figure_intervention_controls,
    "intervention_quality": figure_intervention_quality,
    "sae_diagnostics": figure_sae_diagnostics,
    "sae_features": figure_sae_features,
}


def build_all_figures(
    output_dir: str | Path, figures_dir: str | Path | None = None,
    *, only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate every figure whose source artifact exists, and say what did not.

    Returns a manifest that records, per figure, the artifact it came from and
    that artifact's SHA-256, so the audit can verify no figure is orphaned.
    """
    root = Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment output directory is missing: {root}")
    figures_root = Path(figures_dir) if figures_dir is not None else root / "figures"
    artifacts = load_result_artifacts(root)
    if not artifacts["present"]:
        raise ResearchIntegrityError(
            f"No result artifacts found in {root}. Figures are generated only from real "
            "results; there is no sample-data path.")
    context = build_figure_context(artifacts)

    requested = list(only) if only else list(FIGURE_BUILDERS)
    unknown = [name for name in requested if name not in FIGURE_BUILDERS]
    if unknown:
        raise ResearchIntegrityError(f"Unknown figure names: {unknown}")

    figures: dict[str, Any] = {}
    for name in requested:
        figures[name] = FIGURE_BUILDERS[name](artifacts, context, figures_root)

    manifest = {
        "experiment": context.to_dict(),
        "source_artifacts": artifacts["present"],
        "source_checksums": artifacts["checksums"],
        "figures_dir": str(figures_root),
        "figures": figures,
        "n_generated": sum(1 for f in figures.values() if f["status"] == "ok"),
        "n_not_run": sum(1 for f in figures.values() if f["status"] == "not_run"),
        "not_run": {n: f["reason"] for n, f in figures.items() if f["status"] == "not_run"},
    }
    figures_root.mkdir(parents=True, exist_ok=True)
    (figures_root / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit_figures(figures_dir: str | Path) -> list[str]:
    """Verify every recorded figure exists and points at a real artifact."""
    root = Path(figures_dir)
    manifest_path = root / "figures_manifest.json"
    if not manifest_path.is_file():
        return [f"missing figures manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, entry in (manifest.get("figures") or {}).items():
        if entry.get("status") != "ok":
            continue
        for key in ("figure", "source_data"):
            if not entry.get(key) or not Path(entry[key]).is_file():
                failures.append(f"figure {name}: {key} is missing from disk")
        source = entry.get("generated_from")
        if not source:
            failures.append(f"figure {name} does not record the artifact it came from")
        elif not Path(source).is_file():
            failures.append(f"figure {name} points at a missing artifact: {source}")
        elif entry.get("source_sha256") and _sha256(Path(source)) != entry["source_sha256"]:
            failures.append(f"figure {name} is stale: {source} changed since it was generated")
        if not entry.get("caption"):
            failures.append(f"figure {name} has no caption metadata")
    return failures


__all__ = [
    "FIGURE_BUILDERS", "FigureContext", "PALETTE", "audit_figures", "build_all_figures",
    "build_figure_context", "load_result_artifacts",
]
