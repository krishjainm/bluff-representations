"""
End-to-end **paper suite**: optional balancing, probe pipeline, cross-context
steering on the subject LM, and causal control grid. Writes JSON artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .behavioral_metrics import score_response_quality, deception_proxy_factual
from .causal_generation import CausalLMInterventionRunner
from .data_loader import DeceptionDataLoader
from .dataset_balance import balance_base_items_stratified
from .paper_controls import control_suite_to_jsonable, run_causal_control_suite
from .training_pipeline import DeceptionTrainingPipeline

logger = logging.getLogger(__name__)


def _pick_steering_for_scenario(
    X: torch.Tensor, y: torch.Tensor
) -> Tuple[int, torch.Tensor, float]:
    """Best layer by train AUC on activations; return layer idx, unit vector, auc."""
    yv = y.detach().cpu().numpy().astype(int)
    if len(np.unique(yv)) < 2:
        w = np.ones(X.shape[2], dtype=np.float32)
        w /= np.linalg.norm(w) + 1e-12
        return 0, torch.tensor(w), 0.5
    best_auc, best_li, best_w = 0.5, 0, None
    for li in range(X.shape[1]):
        Xl = X[:, li, :].detach().cpu().numpy()
        if len(np.unique(yv)) < 2:
            continue
        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"
        )
        clf.fit(Xl, yv)
        p = clf.predict_proba(Xl)[:, 1]
        try:
            auc = float(roc_auc_score(yv, p))
        except Exception:
            auc = 0.5
        w = clf.coef_.ravel().astype(np.float32)
        w = w / (np.linalg.norm(w) + 1e-12)
        if auc > best_auc:
            best_auc, best_li, best_w = auc, li, w
    if best_w is None:
        best_w = np.ones(X.shape[2], dtype=np.float32)
        best_w /= np.linalg.norm(best_w)
    return best_li, torch.tensor(best_w, dtype=torch.float32), best_auc


def run_paper_full_suite(
    csv_path: str,
    output_dir: str,
    activation_dir: Optional[str] = None,
    device: str = "cpu",
    hf_subject_model: Optional[str] = None,
    balance_dataset: bool = True,
    run_probes: bool = True,
    run_cross_context_steering: bool = False,
    run_causal_controls: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.1,
    max_rows_per_steering_cell: int = 12,
    prompt_template: str = (
        "### Instruction:\nAnswer clearly.\n\n### Input:\n{statement}\n\n### Response:\n"
    ),
    steering_strength: float = 1.0,
    intervention_site: str = "block",
    control_suite_strengths: Optional[List[float]] = None,
    causal_max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """
    Args:
        hf_subject_model: Required if ``run_cross_context_steering`` or
            ``run_causal_controls`` is True.
    """
    out: Dict[str, Any] = {"stages": {}}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DeceptionDataLoader(device=device)
    loader.load_csv(csv_path, activation_dir=activation_dir)
    df = loader.data.copy()
    balanced_path = str(csv_path)
    if balance_dataset:
        df = balance_base_items_stratified(df)
        balanced_path = str(out_dir / "balanced_input.csv")
        df.to_csv(balanced_path, index=False)
        out["stages"]["balance"] = {"rows_after": len(df), "csv": balanced_path}
    loader.data = df

    if run_probes:
        pipe = DeceptionTrainingPipeline(device=device, output_dir=str(out_dir / "probes"))
        probe_res = pipe.run_full_experiment(
            csv_path=balanced_path,
            activation_dir=activation_dir,
            split_mode="base_item",
            test_size=test_size,
            val_size=val_size,
            save_results=True,
        )
        out["stages"]["probes"] = {
            "output": str(out_dir / "probes"),
            "best_auc": probe_res["probe_results"]["summary"].get("best_auc"),
            "best_ece": probe_res["probe_results"]["summary"].get("best_ece"),
        }

    train_pack, val_pack, test_pack = loader.split_by_base_item(
        test_size=test_size, val_size=val_size if val_size > 0 else 0.0
    )
    train_df = train_pack["dataframe"]
    test_df = test_pack["dataframe"]

    scenarios = sorted(train_df["scenario"].dropna().unique().tolist())

    if run_cross_context_steering:
        if not hf_subject_model:
            raise ValueError("hf_subject_model required for cross-context steering")
        runner = CausalLMInterventionRunner(
            hf_subject_model, device=device, intervention_site=intervention_site
        )
        steering_table: Dict[str, Any] = {}
        for source in scenarios:
            tr_sub = train_df[train_df["scenario"] == source]
            if len(tr_sub) < 6:
                continue
            Xs = loader.get_activations_tensor(tr_sub)
            ys = loader.get_labels_tensor(tr_sub)
            li, vec, tr_auc = _pick_steering_for_scenario(Xs, ys)
            steering_table[source] = {
                "train_auc_direction": tr_auc,
                "layer_idx": li,
                "targets": {},
            }
            for target in scenarios:
                te_sub = test_df[test_df["scenario"] == target].head(
                    max_rows_per_steering_cell
                )
                rows_out = []
                for _, row in te_sub.iterrows():
                    prompt = prompt_template.format(statement=str(row["statement"]))
                    base = runner.generate(
                        prompt, max_new_tokens=causal_max_new_tokens, strength=0.0
                    )
                    steered = runner.generate(
                        prompt,
                        max_new_tokens=causal_max_new_tokens,
                        layer_idx=li,
                        steering_vector=vec,
                        strength=steering_strength,
                        steering_mode="add",
                    )
                    bq_b = score_response_quality(base.text)
                    bq_s = score_response_quality(steered.text)
                    rec: Dict[str, Any] = {
                        "sample_id": int(row.get("sample_id", -1)),
                        "baseline": base.text[:2000],
                        "steered": steered.text[:2000],
                        "quality_baseline": bq_b.__dict__,
                        "quality_steered": bq_s.__dict__,
                    }
                    if "ground_truth" in row.index and pd.notna(row.get("ground_truth")):
                        rec["behavior"] = deception_proxy_factual(
                            steered.text,
                            str(row["ground_truth"]),
                            int(row["label"]),
                        )
                    rows_out.append(rec)
                steering_table[source]["targets"][target] = rows_out
        path = out_dir / "cross_context_steering.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(steering_table, f, indent=2)
        out["stages"]["cross_context_steering"] = {"file": str(path)}

    if run_causal_controls:
        if not hf_subject_model:
            raise ValueError("hf_subject_model required for causal controls")
        runner = CausalLMInterventionRunner(
            hf_subject_model, device=device, intervention_site=intervention_site
        )
        if len(train_df) < 4:
            raise ValueError("Need more training rows for control suite")
        Xtr = loader.get_activations_tensor(train_df)
        ytr = loader.get_labels_tensor(train_df)
        li, probe_vec, _ = _pick_steering_for_scenario(Xtr, ytr)
        row0 = train_df.iloc[0]
        row1 = train_df.iloc[min(1, len(train_df) - 1)]
        p0 = prompt_template.format(statement=str(row0["statement"]))
        p1 = prompt_template.format(statement=str(row1["statement"]))
        h_mis = runner.last_hidden_at_layer(p1, li)
        suite = run_causal_control_suite(
            runner,
            p0,
            li,
            probe_vec,
            mismatched_patch_vector=h_mis,
            strengths=control_suite_strengths or [0.0, 1.0],
            max_new_tokens=causal_max_new_tokens,
        )
        path = out_dir / "causal_control_suite.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(control_suite_to_jsonable(suite), f, indent=2)
        out["stages"]["causal_control_suite"] = {"file": str(path)}

    summary_path = out_dir / "paper_suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    out["summary_path"] = str(summary_path)

    if run_probes:
        probe_dir = out_dir / "probes"
        if (probe_dir / "experiment_results.json").exists():
            try:
                from .paper_figure_suite import build_paper_figures_from_experiment_dir

                out["stages"]["paper_figures"] = build_paper_figures_from_experiment_dir(
                    probe_dir, auxiliary_dir=out_dir
                )
            except Exception as e:
                out["stages"]["paper_figures_error"] = str(e)

    return out
