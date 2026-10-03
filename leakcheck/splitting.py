"""설계서의 분할 방식대로 예측 행을 학습·시험(또는 폴드)에 배정한다 (Q2 데이터 단계 점검용)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from leakcheck import rules

ROW_LEVEL_METHODS = {"random_rows", "temporal"}


def effective_key_level(split: dict) -> str:
    """실제로 함께 움직이는 묶음. 행 단위 방법이면 키와 무관하게 index_row."""
    if split.get("method") in ROW_LEVEL_METHODS:
        return "index_row"
    return rules.split_level(split["key"])


def group_values(index_rows: pd.DataFrame, level: str) -> pd.Series:
    if level == "family":
        return index_rows["family_id"].astype("string").fillna(index_rows["patient_id"].astype("string"))
    return index_rows[rules.SPLIT_KEY_COLUMN[level]].astype("string")


def assign(index_rows: pd.DataFrame, split: dict) -> pd.Series:
    """예측 행 → 배정 이름 (test/train 또는 fold0..k-1). index는 index_rows와 같다."""
    seed = split.get("seed", 0)
    method = split.get("method", "group_holdout")
    if method == "temporal":
        frac = split.get("test_fraction", 0.3)
        cut = index_rows["tp"].quantile(1 - frac)
        return pd.Series(np.where(index_rows["tp"] > cut, "test", "train"), index=index_rows.index)
    level = effective_key_level(split)
    groups = group_values(index_rows, level)
    uniq = np.array(sorted(groups.unique()))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    if method == "group_kfold":
        k = split.get("n_folds", 5)
        lab = {g: f"fold{i % k}" for i, g in zip(range(len(uniq)), uniq[perm])}
    else:
        n_test = int(round(split.get("test_fraction", 0.3) * len(uniq)))
        test = set(uniq[perm[:n_test]])
        lab = {g: ("test" if g in test else "train") for g in uniq}
    return groups.map(lab)


def crossing(index_rows: pd.DataFrame, labels: pd.Series, level: str) -> tuple[int, int]:
    """level 묶음 중 둘 이상의 배정에 나뉜 것의 수, 전체 묶음 수. 가족은 실제 가족만 센다."""
    if level == "family":
        mask = index_rows["family_id"].notna()
        g = index_rows.loc[mask, "family_id"].astype("string")
        lab = labels[mask]
    else:
        g = group_values(index_rows, level)
        lab = labels
    n = pd.DataFrame({"g": g, "lab": lab}).groupby("g")["lab"].nunique()
    return int((n > 1).sum()), int(len(n))
