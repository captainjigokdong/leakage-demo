"""결과 정의. 결정 카드의 결과율(기술 통계)에 쓴다. 모델 성능과는 무관하다.

- next_admission   결과 창 안에 같은 환자의 입원이 시작됨 (30일 재입원 등)
- kdigo_creatinine 결과 창 안에 채취된 크레아티닌이 KDIGO 기준을 처음 만족
                   (앞선 48시간 안 측정보다 0.3 mg/dL 이상, 또는 앞선 7일 안 최솟값의 1.5배 이상)
                   비교 대상 측정은 reference_window ∪ 결과 창 안의 것만 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from leakcheck import timeline
from leakcheck.features import _apply_filter, make_feature
from synth.stats import EPS, KDIGO_ABS_RISE, KDIGO_ABS_WINDOW_H, KDIGO_REL_RISE, KDIGO_REL_WINDOW_H


def _kdigo_event(h: np.ndarray, v: np.ndarray, in_window: np.ndarray) -> bool:
    for j in np.flatnonzero(in_window):
        prior = np.arange(len(v)) < j
        rel = prior & (h >= h[j] - KDIGO_REL_WINDOW_H)
        if rel.any() and v[j] >= KDIGO_REL_RISE * v[rel].min() - EPS:
            return True
        ab = prior & (h >= h[j] - KDIGO_ABS_WINDOW_H)
        if ab.any() and v[j] - v[ab].min() >= KDIGO_ABS_RISE - EPS:
            return True
    return False


def label(design: dict, tagged: dict[str, pd.DataFrame], index_rows: pd.DataFrame) -> pd.Series:
    """예측 행마다 결과 (index_id → bool)."""
    o = design["outcome"]
    ids = index_rows["index_id"]
    if o["definition"] == "next_admission":
        spec = {"name": o["name"], "source": "admissions", "scope": "patient_history", "filter": o["filter"],
                "time_column": o.get("time_column", "admit_time"), "window": o["window"], "agg": "any"}
        r = make_feature(tagged, spec, index_rows)
        return r.frame.set_index("index_id")["value"].astype(bool)

    if o["definition"] == "kdigo_creatinine":
        tc = o.get("time_column", "collect_time")
        labs = _apply_filter(tagged["labs"], o["filter"])
        m = index_rows[["index_id", "admission_id"]].merge(
            labs[["_admission_id", tc, "value"]], left_on="admission_id", right_on="_admission_id")
        ir = index_rows.set_index("index_id")
        ref = o.get("reference_window", {})
        lo = timeline.resolve(ref.get("start", o["window"].get("start", "-inf")), ir).reindex(m["index_id"])
        ws = timeline.resolve(o["window"].get("start", "-inf"), ir).reindex(m["index_id"])
        we = timeline.resolve(o["window"].get("end", "inf"), ir).reindex(m["index_id"])
        t = m[tc].to_numpy()
        m = m.assign(_in=(t > ws.to_numpy()) & (t <= we.to_numpy()))
        m = m[(t > lo.to_numpy()) & (t <= we.to_numpy())].sort_values(["index_id", tc])
        out = {}
        for iid, g in m.groupby("index_id", sort=False):
            if not g["_in"].any():
                continue
            h = (g[tc] - g[tc].iloc[0]).dt.total_seconds().to_numpy() / 3600
            out[iid] = _kdigo_event(h, g["value"].to_numpy(dtype=float), g["_in"].to_numpy())
        return pd.Series(out, dtype=bool).reindex(ids, fill_value=False).set_axis(ids)

    raise ValueError(f"알 수 없는 결과 정의: {o['definition']}")
