"""출처를 기록하는 특징 생성 (CLAUDE.md 절대 규칙 5).

특징은 make_feature()로만 만든다. 결과 특징은 재료 행의 꼬리표를 물려받는다.
- available_time = 재료 행들의 확인 가능 시각 최댓값 (창의 끝이 유한하면 그것도 포함:
  "창 안에 아무 일도 없었다"는 사실은 창이 닫혀야 알 수 있다)
- provenance     = 재료 행 ID 목록 + 계산 방식

FeatureRegistry에 등록되지 않은 열은 "출처 불명"이다 (Q1~Q3 확인 불가).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leakcheck import rules, timeline
from leakcheck.tagging import available_series
from synth.stats import kdigo_first

UNKNOWN = "출처 불명"
INDEX_COLUMNS = ["index_id", "admission_id", "patient_id", "family_id", "unit",
                 "admit_time", "discharge_time", "tp", "landmark_h"]


def build_index(design: dict, tagged: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """예측 행. 고정: 입원당 하나 / 동적: 입원 × 랜드마크 (tₚ < 퇴원인 것만)."""
    adm = tagged["admissions"]
    fam = tagged["patients"].set_index("patient_id")["family_id"]
    anchor = design["tp"]["anchor"]
    parts = []
    for o in design["tp"]["offsets_h"]:
        base = adm["admit_time"] if anchor == "admit" else adm["discharge_time"]
        tp = base + pd.Timedelta(hours=o)
        keep = tp < adm["discharge_time"] if anchor == "admit" else pd.Series(True, index=adm.index)
        a = adm[keep]
        parts.append(pd.DataFrame({
            "index_id": a["admission_id"] + f"@{o:g}h",
            "admission_id": a["admission_id"],
            "patient_id": a["patient_id"],
            "family_id": a["patient_id"].map(fam),
            "unit": a["unit"],
            "admit_time": a["admit_time"],
            "discharge_time": a["discharge_time"],
            "tp": tp[keep],
            "landmark_h": float(o),
        }))
    idx = pd.concat(parts, ignore_index=True)
    return idx.sort_values(["admission_id", "landmark_h"], kind="stable").reset_index(drop=True)


@dataclass
class FeatureResult:
    name: str
    frame: pd.DataFrame   # index_id, value, available_time, provenance
    method: str
    spec: dict

    def leaks(self, index_rows: pd.DataFrame, ref: str = "tp") -> pd.Series:
        """확인 가능 시각이 기준 시점(기본 tₚ)보다 늦은 예측 행."""
        ref_t = timeline.resolve(ref, index_rows).set_axis(index_rows["index_id"])
        at = self.frame.set_index("index_id")["available_time"].reindex(ref_t.index)
        return (at > ref_t).fillna(False)


@dataclass
class FeatureRegistry:
    """make_feature로 만든 특징의 출처 기록. 여기 없는 열은 출처 불명."""
    entries: dict[str, FeatureResult] = field(default_factory=dict)

    def add(self, r: FeatureResult) -> None:
        self.entries[r.name] = r

    def provenance_of(self, column: str) -> str:
        r = self.entries.get(column)
        return r.method if r else UNKNOWN

    def unknown_columns(self, columns) -> list[str]:
        return [c for c in columns if c not in self.entries]


def describe(spec: dict) -> str:
    """계산 방식 문자열 (provenance의 일부)."""
    col = spec.get("column") or rules.table_rule(spec["source"]).value_column or "*"
    flt = ", ".join(f"{k}∈{{{','.join(map(str, v))}}}" for k, v in spec.get("filter", {}).items())
    w = spec.get("window")
    win = ""
    if w:
        tc = spec.get("time_column") or rules.default_time_column(spec["source"])
        win = f", {tc}∈({w.get('start', '-inf')}, {w.get('end', 'inf')}]"
    return f"{spec['agg']}({spec['source']}.{col}{' | ' + flt if flt else ''}{win}, scope={spec['scope']})"


def _scope_pairs(index_rows: pd.DataFrame, adm: pd.DataFrame, scope: str) -> pd.DataFrame:
    """예측 행 → 재료 입원 (index_id, src_admission_id)."""
    if scope == "index_admission":
        return pd.DataFrame({"index_id": index_rows["index_id"], "src_admission_id": index_rows["admission_id"]})
    m = index_rows[["index_id", "patient_id", "admission_id", "admit_time"]].merge(
        adm[["patient_id", "admission_id", "admit_time", "discharge_time"]].rename(
            columns={"admission_id": "src_admission_id", "admit_time": "src_admit", "discharge_time": "src_discharge"}),
        on="patient_id")
    if scope == "prior_admissions":
        m = m[m["src_discharge"] <= m["admit_time"]]
    elif scope == "next_admission":
        m = m[m["src_admit"] > m["admit_time"]].sort_values(["index_id", "src_admit"])
        m = m.groupby("index_id", sort=False).head(1)
    elif scope != "patient_history":
        raise ValueError(f"알 수 없는 범위: {scope}")
    return m[["index_id", "src_admission_id"]]


def _apply_filter(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    for col, vals in flt.items():
        if col not in df:
            raise rules.RuleMissing(f"필터 열 '{col}'이 테이블에 없다")
        if col in rules.CODE_COLUMNS:
            df = df[df[col].astype("string").str.startswith(tuple(str(v) for v in vals)).fillna(False)]
        else:
            df = df[df[col].isin(vals)]
    return df


def _time_values(df: pd.DataFrame, table: str, col: str) -> pd.Series:
    kind = rules.table_rule(table).time_columns.get(col)
    if kind is None:
        raise rules.RuleMissing(f"'{table}.{col}'은 규칙표의 시각 열이 아니다")
    if kind == "date":
        return pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    return df[col]


def _groups(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """정렬된 id 배열 → (각 묶음의 id, 경계 위치)."""
    if len(ids) == 0:
        return ids, np.array([0])
    cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    return ids[np.r_[0, cut]], np.r_[0, cut, len(ids)]


def _kdigo_by_group(m: pd.DataFrame) -> pd.Series:
    m = m.sort_values(["index_id", "collect_time"], kind="stable")
    keys, b = _groups(m["index_id"].to_numpy())
    h = m["collect_time"].to_numpy().astype("datetime64[m]").astype(np.int64) / 60
    v = m["val"].to_numpy(dtype=float)
    out = [kdigo_first(h[b[i]:b[i + 1]], v[b[i]:b[i + 1]]) is not None for i in range(len(keys))]
    return pd.Series(out, index=keys, dtype=bool)


def _provenance_by_group(m: pd.DataFrame) -> pd.Series:
    keys, b = _groups(m["index_id"].to_numpy())
    p = m["_provenance"].to_numpy()
    return pd.Series([tuple(p[b[i]:b[i + 1]]) for i in range(len(keys))], index=keys, dtype=object)


def make_feature(tagged: dict[str, pd.DataFrame], spec: dict, index_rows: pd.DataFrame) -> FeatureResult:
    """설계서의 특징(또는 포함·제외 기준) 명세 하나를 예측 행마다 계산한다."""
    if "source" not in spec:
        raise rules.RuleMissing(f"특징 '{spec.get('name')}'에 source가 없다 ({UNKNOWN})")
    table = spec["source"]
    rule = rules.table_rule(table)
    src = tagged[table]
    scope = spec["scope"]
    col = spec.get("column") or rule.value_column
    agg = spec["agg"]
    if col is not None and col not in src:
        raise rules.RuleMissing(f"'{table}'에 열 '{col}'이 없다")

    if table == "patients":
        m = index_rows[["index_id", "patient_id"]].merge(src, on="patient_id")
    else:
        pairs = _scope_pairs(index_rows, tagged["admissions"], scope)
        key = "admission_id" if table == "admissions" else "_admission_id"
        m = pairs.merge(src, left_on="src_admission_id", right_on=key)
    m = _apply_filter(m, spec.get("filter", {}))

    m = m.assign(_avail=available_series(m, table, spec.get("column")).to_numpy())
    tc = spec.get("time_column") or rules.default_time_column(table)
    w = spec.get("window")
    if w:
        if tc is None:
            raise rules.RuleMissing(f"'{table}'에는 창을 적용할 시각 열이 없다")
        ir = index_rows.set_index("index_id")
        t = _time_values(m, table, tc)
        start = timeline.resolve(w.get("start", "-inf"), ir).reindex(m["index_id"]).to_numpy()
        end = timeline.resolve(w.get("end", "inf"), ir).reindex(m["index_id"]).to_numpy()
        m = m[(t.to_numpy() > start) & (t.to_numpy() <= end)]
    order = _time_values(m, table, tc) if tc else m["_avail"]
    m = m.assign(_order=order.to_numpy(), val=m[col].to_numpy() if col else np.nan)
    m["val_num"] = pd.to_numeric(m["val"], errors="coerce")

    m = m.sort_values(["index_id", "_order"], kind="stable")
    g = m.groupby("index_id", sort=False)
    if agg == "count":
        val = g.size()
    elif agg == "any":
        val = g.size() > 0
    elif agg in ("value", "last"):
        val = g["val"].last()
    elif agg == "first":
        val = g["val"].first()
    elif agg in ("max", "min", "mean"):
        val = getattr(g["val_num"], agg)()
    elif agg == "kdigo_aki":
        val = _kdigo_by_group(m)
    else:
        raise ValueError(f"알 수 없는 집계: {agg}")

    ids = index_rows["index_id"]
    out = pd.DataFrame({"index_id": ids})
    v = val.reindex(ids)
    if agg == "count":
        v = v.fillna(0).astype(int)
    elif agg in ("any", "kdigo_aki"):
        v = v.astype("boolean").fillna(False).astype(bool)
    out["value"] = v.to_numpy()
    avail = g["_avail"].max().reindex(ids)
    if w and timeline.parse(w.get("end", "inf")).anchor != "inf":
        wend = timeline.resolve(w["end"], index_rows).set_axis(ids)
        avail = pd.concat([avail, wend], axis=1).max(axis=1)
    out["available_time"] = avail.to_numpy()
    prov = _provenance_by_group(m).reindex(ids)
    out["provenance"] = [p if isinstance(p, tuple) else () for p in prov]
    return FeatureResult(spec["name"], out, describe(spec), spec)


def feature_matrix(results: list[FeatureResult], index_rows: pd.DataFrame) -> tuple[pd.DataFrame, FeatureRegistry]:
    reg = FeatureRegistry()
    X = index_rows[["index_id"]].copy()
    for r in results:
        reg.add(r)
        X[r.name] = r.frame.set_index("index_id")["value"].reindex(X["index_id"]).to_numpy()
    return X, reg
