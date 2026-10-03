"""누수 효과 시연 (7단계 보조 분석, docs/step7_analysis_plan.md 4절).

같은 합성 데이터에서 수정 설계(기본 설계서, 점검기 통과)와 누수 설계(공개 오류 목록의 누수를 하나씩 / 모두 넣은 것)를
학습해 시험 집합 AUROC를 비교한다. 결과는 절대값이 아니라 누수 유무에 따른 차이로 해석한다.

- 특징: leakcheck.features.make_feature (출처 기록 함수)만 쓴다. 만들 수 없는 특징은 빼고 기록한다.
- 결과: leakcheck.outcomes.label / 분할: leakcheck.splitting.assign
- 성능: 모든 설계를 결정 카드 → 승인 → 잠금 뒤 leakcheck.lock.auroc로만 계산한다.
- TPOT 1.1.0: fit(X, y)가 그룹을 받지 않고 cv=정수이면 행 단위 StratifiedKFold를 쓴다.
  그래서 학습 집합 안에서 분할 단위(가족, 없으면 환자)로 미리 나눈 폴드를 넘기고, 실제로 그 폴드가 쓰였는지 기록한다.

  python -m experiment.leakage_effect [--no-tpot] [--seeds 5]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression

from leakcheck import design as dz
from leakcheck import lock as lk
from leakcheck import outcomes, rules, splitting
from leakcheck.checks import prepare_data
from leakcheck.features import feature_matrix, make_feature

ROOT = Path(__file__).resolve().parents[1]
BASES = {"fixed": ROOT / "designs" / "base" / "fixed_readmission.json",
         "dynamic": ROOT / "designs" / "base" / "dynamic_aki.json"}
OUT = ROOT / "results"
APPROVER = "사용자 (7단계 계획 승인 2026-10-03, 누수 효과 시연용 설계 일괄 승인)"
SEED_STEPS = range(5)
TPOT_MINUTES = 5
TPOT_FOLDS = 5


# ---------------------------------------------------------------- 누수 목록 (docs/step7_analysis_plan.md 4.1, 실행 전 고정)

def _feat(d: dict, spec: dict) -> None:
    d["features"].append(spec)


def _fit_all(d: dict, name: str) -> None:
    for p in d["preprocessing"]:
        if p["name"] == name:
            p["fit_scope"] = "all"


def _select_all(d: dict) -> None:
    d["preprocessing"].append({"name": "select_k", "kind": "select", "fit_scope": "all"})


LEAKS = {
    "fixed": {
        "F1": ("E12 다음 입원 병동 특징 (Q4)", lambda d: _feat(d, {
            "name": "next_unit", "source": "admissions", "scope": "next_admission", "column": "unit", "agg": "value"})),
        "F2": ("E06 입원 단위 분할 (Q2)", lambda d: d["split"].update(key="admission_id")),
        "F3": ("E08 중앙값 대치를 전체 데이터로 적합 (Q3)", lambda d: _fit_all(d, "median_impute")),
        "F4": ("E10 상위 k개 특징 선택을 전체 데이터로 적합 (Q3)", _select_all),
    },
    "dynamic": {
        "D1": ("E01 이번 입원 AKI 진단 코드 N17 (Q1)", lambda d: _feat(d, {
            "name": "dx_aki", "source": "diagnoses", "scope": "index_admission", "filter": {"icd_code": ["N17"]},
            "agg": "any"})),
        "D2": ("E03 입원 전체 기간 최고 크레아티닌 (Q1)", lambda d: _feat(d, {
            "name": "cr_max_adm", "source": "labs", "filter": {"test": ["creatinine"]}, "time_column": "report_time",
            "window": {"start": "admit", "end": "discharge"}, "agg": "max"})),
        "D3": ("행(예측 시점) 단위 분할 (Q2)", lambda d: d["split"].update(method="random_rows")),
        "D4": ("E08 중앙값 대치를 전체 데이터로 적합 (Q3)", lambda d: _fit_all(d, "median_impute")),
        "D5": ("E10 상위 k개 특징 선택을 전체 데이터로 적합 (Q3)", _select_all),
        "D6": ("E13 투석 오더·신장내과 협진 오더 (Q4, 대리 변수)", lambda d: _feat(d, {
            "name": "renal_order", "source": "orders", "filter": {"order_type": ["dialysis_order", "nephrology_consult"]},
            "time_column": "order_time", "window": {"start": "admit", "end": "tp"}, "agg": "any"})),
    },
}


def variants(t: str) -> dict[str, tuple[str, dict]]:
    """이름 → (설명, 설계서). fixed = 수정 설계(기본), 누수 하나씩, all = 그 유형의 누수 모두."""
    base = json.loads(BASES[t].read_text(encoding="utf-8"))
    out = {"fixed": ("수정 설계 (기본 설계서, 점검기 통과)", base)}
    every = copy.deepcopy(base)
    for lid, (desc, fn) in LEAKS[t].items():
        d = copy.deepcopy(base)
        fn(d)
        d["design_id"] = f"{base['design_id']}+{lid}"
        out[lid] = (desc, d)
        fn(every)
    every["design_id"] = f"{base['design_id']}+all"
    out["all"] = ("누수 모두", every)
    return out


def with_seed(d: dict, step: int) -> dict:
    d = copy.deepcopy(d)
    d["split"]["seed"] = d["split"].get("seed", 0) + step
    return d


# ---------------------------------------------------------------- 자료 · 전처리

def build(d: dict, tables: dict) -> dict:
    """설계서 → 예측 행, 결과, 특징 행렬(출처 기록 함수로만), 만들 수 없던 특징."""
    nd = dz.normalize(d)
    ctx = prepare_data(nd, tables)
    rows = ctx.cohort
    y = outcomes.label(nd, ctx.tagged, rows).reindex(rows["index_id"]).to_numpy().astype(int)
    results, skipped = [], {}
    for spec in nd["features"]:
        try:
            results.append(make_feature(ctx.tagged, spec, rows))
        except rules.RuleMissing as e:
            skipped[spec["name"]] = str(e)
    X, reg = feature_matrix(results, rows)
    X = X.drop(columns="index_id")
    assert not reg.unknown_columns(X.columns), "출처 불명 열"
    return {"design": nd, "rows": rows, "y": y, "X": X, "skipped": skipped}


def _scope_mask(scope: str, train: np.ndarray) -> np.ndarray:
    return np.ones_like(train) if scope == "all" else train


def preprocess(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, steps: list[dict]) -> tuple[np.ndarray, list[str]]:
    """설계서의 전처리를 fit_scope대로 적용한다 (train: 학습 행만, all: 시험 행 포함)."""
    X = X.copy()
    scope = {p["kind"]: p.get("fit_scope", "train") for p in steps}
    cat = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
    num = [c for c in X.columns if c not in cat]
    X[num] = X[num].astype(float)
    if "impute" in scope:
        med = X.loc[_scope_mask(scope["impute"], train), num].median()
        X[num] = X[num].fillna(med).fillna(0.0)
    for c in cat:
        X[c] = X[c].astype("string").fillna("missing")
    fit = _scope_mask(scope.get("encode", "train"), train)
    parts, names = [X[num].to_numpy(float)], list(num)
    for c in cat:
        levels = sorted(X.loc[fit, c].unique())
        parts.append(np.column_stack([(X[c] == lv).to_numpy(float) for lv in levels]))
        names += [f"{c}={lv}" for lv in levels]
    Z = np.column_stack(parts)
    if "scale" in scope:
        m = _scope_mask(scope["scale"], train)
        mu, sd = Z[m].mean(0), Z[m].std(0)
        Z = (Z - mu) / np.where(sd > 0, sd, 1.0)
    if "select" in scope:
        m = _scope_mask(scope["select"], train)
        k = math.ceil(Z.shape[1] / 2)
        f, _ = f_classif(Z[m], y[m])
        keep = np.sort(np.argsort(-np.nan_to_num(f))[:k])
        Z, names = Z[:, keep], [names[i] for i in keep]
    return Z, names


# ---------------------------------------------------------------- 환자 단위 폴드 (TPOT 우회)

class GroupFolds:
    """미리 나눈 그룹 폴드. TPOT가 cv로 받으면 sklearn check_cv가 그대로 쓴다.
    split 호출 수를 클래스 수준에서 센다 (TPOT/dask가 객체를 복사해 써도 같은 프로세스 안이면 남는다)."""
    total_calls = 0

    def __init__(self, groups: np.ndarray, k: int, seed: int):
        uniq = np.array(sorted(set(groups)))
        perm = np.random.default_rng(seed).permutation(len(uniq))
        fold_of = {g: i % k for i, g in zip(range(len(uniq)), uniq[perm])}
        self.fold = np.array([fold_of[g] for g in groups])
        self.groups, self.k, self.calls = groups, k, 0

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.k

    def split(self, X=None, y=None, groups=None):
        self.calls += 1
        type(self).total_calls += 1
        for i in range(self.k):
            yield np.flatnonzero(self.fold != i), np.flatnonzero(self.fold == i)

    def crossing_groups(self) -> int:
        return int(pd.DataFrame({"g": self.groups, "f": self.fold}).groupby("g")["f"].nunique().gt(1).sum())


def row_kfold_crossing(groups: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict:
    """TPOT 기본(cv=정수 → StratifiedKFold(shuffle))이었다면 같은 그룹이 여러 폴드에 나뉘는 정도."""
    from sklearn.model_selection import StratifiedKFold
    fold = np.empty(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        fold[te] = i
    n = pd.DataFrame({"g": groups, "f": fold}).groupby("g")["f"].nunique()
    multi = pd.Series(groups).value_counts()
    return {"groups": int(len(n)), "groups_with_multiple_rows": int((multi > 1).sum()), "groups_split_across_folds": int((n > 1).sum())}


# ---------------------------------------------------------------- 실행

def models(seed: int) -> dict:
    return {"logistic": LogisticRegression(max_iter=3000),
            "hgb": HistGradientBoostingClassifier(random_state=seed)}


def tpot_model(folds: GroupFolds, seed: int):
    from tpot import TPOTClassifier
    return TPOTClassifier(search_space="linear", scorers=["roc_auc"], scorers_weights=[1], cv=folds,
                          generations=5, population_size=12, max_time_mins=TPOT_MINUTES, max_eval_time_mins=1,
                          n_jobs=4, processes=False, random_state=seed, verbose=0)


def run_one(t: str, name: str, d: dict, tables: dict, step: int, use_tpot: bool) -> dict:
    d = with_seed(d, step)
    card = lk.decision_card(d, tables)
    lock = lk.DecisionLock(card)
    lock.approve(APPROVER)
    b = build(d, tables)
    split = splitting.assign(b["rows"], b["design"]["split"])
    train = (split == "train").to_numpy()
    Z, names = preprocess(b["X"], b["y"], train, b["design"]["preprocessing"])
    level = splitting.effective_key_level(b["design"]["split"])
    crossing, n_groups = splitting.crossing(b["rows"], split, rules.split_level(BASE_SPLIT_KEY[t]))
    rec = {"type": t, "design": name, "seed_step": step, "split_seed": b["design"]["split"]["seed"],
           "split_level": level, "n_rows": int(len(b["y"])), "n_train": int(train.sum()), "n_test": int((~train).sum()),
           "outcome_rate": float(b["y"].mean()), "n_columns": len(names), "skipped_features": b["skipped"],
           "patients_or_families_in_both": crossing, "auroc": {}}
    for mname, m in models(b["design"]["split"]["seed"]).items():
        m.fit(Z[train], b["y"][train])
        rec["auroc"][mname] = lk.auroc(lock, d, b["y"][~train], m.predict_proba(Z[~train])[:, 1])
    if use_tpot:
        grp = splitting.group_values(b["rows"], rules.split_level(BASE_SPLIT_KEY[t])).to_numpy()[train]
        folds = GroupFolds(grp, TPOT_FOLDS, b["design"]["split"]["seed"])
        tp = tpot_model(folds, b["design"]["split"]["seed"])
        t0 = time.time()
        before = GroupFolds.total_calls
        tp.fit(Z[train], b["y"][train])
        rec["auroc"]["tpot"] = lk.auroc(lock, d, b["y"][~train], tp.predict_proba(Z[~train])[:, 1])
        rec["tpot"] = {"seconds": round(time.time() - t0, 1), "cv_split_calls": GroupFolds.total_calls - before,
                       "cv_gen_is_group_folds": isinstance(tp.cv_gen, GroupFolds),
                       "cv_groups_crossing_folds": folds.crossing_groups(), "cv_folds": TPOT_FOLDS,
                       "pipeline": str(tp.fitted_pipeline_)[:600],
                       "default_rowwise_cv_would_split": row_kfold_crossing(grp, b["y"][train], TPOT_FOLDS,
                                                                            b["design"]["split"]["seed"])}
    rec["lock"] = {"approved_by": lock.approved_by, "design_hash": card.design_hash[:12]}
    return rec


# 수정 설계의 분할 키: "학습·시험 양쪽에 들어간 독립 단위" 수는 이 단위로 센다
BASE_SPLIT_KEY = {"fixed": "family_id", "dynamic": "family_id"}


def summarize(recs: list[dict]) -> dict:
    out = {}
    for r in recs:
        for m, a in r["auroc"].items():
            out.setdefault(r["type"], {}).setdefault(r["design"], {}).setdefault(m, []).append(a)
    table = {}
    for t, ds in out.items():
        ref = {m: np.mean(v) for m, v in ds["fixed"].items()}
        for name, ms in ds.items():
            for m, v in ms.items():
                table.setdefault(t, {}).setdefault(name, {})[m] = {
                    "n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                    "diff_vs_fixed": float(np.mean(v) - ref[m]) if m in ref else None}
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-tpot", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(SEED_STEPS))
    ap.add_argument("--out", type=Path, default=OUT / "leakage_effect.json")
    a = ap.parse_args(argv)
    from synth.generate import load
    tables = load()
    recs = []
    for t in ("fixed", "dynamic"):
        for name, (desc, d) in variants(t).items():
            for step in range(a.seeds):
                tpot = (not a.no_tpot) and step == 0 and name in ("fixed", "all")
                r = run_one(t, name, d, tables, step, tpot)
                r["description"] = desc
                recs.append(r)
                print(json.dumps({k: r[k] for k in ("type", "design", "seed_step", "auroc")}, ensure_ascii=False), flush=True)
    res = {"leaks": {t: {k: v[0] for k, v in LEAKS[t].items()} for t in LEAKS}, "runs": recs, "summary": summarize(recs)}
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
