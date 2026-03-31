"""
Balance paired datasets at **base-item** level (paper design).

Equalizes the number of base items per stratum (e.g. scenario × difficulty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Sequence


def balance_base_items_stratified(
    df: pd.DataFrame,
    stratify_columns: Sequence[str] = ("scenario", "difficulty_bucket"),
    base_item_column: str = "base_item_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Subsample base items so each stratum has the same count (the minimum
    count observed across strata).

    Rows without ``base_item_column`` are dropped from the dedup table; ensure
    ``DeceptionDataLoader._ensure_paper_schema`` ran first.
    """
    df = df.copy()
    cols = list(stratify_columns)
    for c in cols:
        if c not in df.columns:
            df[c] = "default"
    bases = df.drop_duplicates(subset=[base_item_column], keep="first")
    bases["_strat"] = bases[cols[0]].astype(str)
    for c in cols[1:]:
        bases["_strat"] = bases["_strat"] + "_" + bases[c].astype(str)
    counts = bases.groupby("_strat")[base_item_column].nunique()
    if len(counts) == 0:
        return df
    m = int(counts.min())
    if m <= 0:
        return df
    rng = np.random.default_rng(random_state)
    keep_ids: List = []
    for strat in counts.index:
        bids = bases.loc[bases["_strat"] == strat, base_item_column].unique()
        if len(bids) <= m:
            keep_ids.extend(bids.tolist())
        else:
            pick = rng.choice(bids, size=m, replace=False)
            keep_ids.extend(pick.tolist())
    out = df[df[base_item_column].isin(keep_ids)].reset_index(drop=True)
    return out
