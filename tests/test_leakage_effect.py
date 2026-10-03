"""누수 효과 시연 시험 (docs/step7_analysis_plan.md 4절)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiment import leakage_effect as le
from leakcheck import lock as lk


def test_leak_list_matches_plan():
    assert set(le.LEAKS["fixed"]) == {"F1", "F2", "F3", "F4"}
    assert set(le.LEAKS["dynamic"]) == {"D1", "D2", "D3", "D4", "D5", "D6"}


@pytest.mark.parametrize("t", ["fixed", "dynamic"])
def test_variants_change_only_their_leak(t):
    vs = le.variants(t)
    base = vs["fixed"][1]
    assert set(vs) == {"fixed", "all", *le.LEAKS[t]}
    for lid in le.LEAKS[t]:
        d = vs[lid][1]
        assert d != base and d["design_id"].endswith(lid)
        assert {k for k in base if base[k] != d[k]} <= {"design_id", "features", "split", "preprocessing"}
    assert le.variants(t)["fixed"][1] == base   # 원본을 바꾸지 않는다


def test_seed_changes_design_hash_so_each_needs_its_own_lock():
    from leakcheck import design as dz
    d = le.variants("fixed")["fixed"][1]
    assert dz.design_hash(le.with_seed(d, 0)) != dz.design_hash(le.with_seed(d, 1))


def test_auroc_requires_lock():
    d = le.variants("fixed")["fixed"][1]
    with pytest.raises(lk.LockError):
        lk.auroc(None, d, [0, 1], [0.1, 0.9])


def test_group_folds_keep_groups_together_and_count_calls():
    g = np.array(["a", "a", "b", "c", "c", "c", "d", "e"])
    f = le.GroupFolds(g, 3, seed=1)
    before = le.GroupFolds.total_calls
    for tr, te in f.split():
        assert not set(g[tr]) & set(g[te])
    assert f.crossing_groups() == 0 and le.GroupFolds.total_calls == before + 1
    import sklearn.model_selection as ms
    assert ms.check_cv(f, np.array([0, 1] * 4), classifier=True) is f   # TPOT가 그대로 쓴다


def test_rowwise_kfold_splits_groups():
    g = np.repeat(np.arange(50), 4)
    y = np.tile([0, 1], 100)
    r = le.row_kfold_crossing(g, y, 5, seed=0)
    assert r["groups_with_multiple_rows"] == 50 and r["groups_split_across_folds"] > 0


def test_preprocess_fit_scope():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 100.0, np.nan, 200.0], "s": ["M", "F", "M", "F", "M", "F"]})
    y = np.array([0, 1, 0, 1, 0, 1])
    train = np.array([True, True, True, False, False, False])
    zt, names = le.preprocess(X, y, train, [{"kind": "impute", "fit_scope": "train"}])
    za, _ = le.preprocess(X, y, train, [{"kind": "impute", "fit_scope": "all"}])
    i = names.index("a")
    assert zt[1, i] == 2.0 and za[1, i] == np.median([1.0, 3.0, 100.0, 200.0])
    zs, ns = le.preprocess(X, y, train, [{"kind": "impute", "fit_scope": "train"}, {"kind": "select", "fit_scope": "all"}])
    assert zs.shape[1] == len(ns) == 2   # 열 3개(a, s=F, s=M) → 올림 절반
