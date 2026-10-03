"""모든 행에 꼬리표 네 개를 붙인다. docs/research_plan.md 3절.

- _entity          이 값이 누구의 것인가 (환자 ID)
- _split_unit      분할할 때 함께 움직여야 하는 묶음 (설계서의 split_unit이 정함, 기본 환자)
- _available_time  이 값이 알려진 가장 이른 시각 (규칙표의 행 규칙)
- _provenance      원본 테이블#행 ID

열마다 다른 확인 가능 시각(예: 입원 행의 퇴원 상태)은 available_series()로 얻는다.
규칙이 없는 테이블, 모순 행(보고 < 채취, 퇴원 < 입원), 빈 시각·ID, 연결 안 되는 행은
추측하지 않고 모아서 TaggingError로 멈춘다.
"""
from __future__ import annotations

import pandas as pd

from leakcheck import rules
from leakcheck.rules import Avail
from leakcheck.timeline import T_MIN

TAG_COLUMNS = ["_entity", "_split_unit", "_available_time", "_provenance"]
MAX_EXAMPLES = 5


class TaggingError(Exception):
    def __init__(self, problems: list[dict]):
        self.problems = problems
        lines = [f"- {p['table']}: {p['problem']} ({p['n_rows']}행, 예: {', '.join(p['examples'])})"
                 for p in problems]
        super().__init__("꼬리표를 붙일 수 없는 행이 있어 멈춘다:\n" + "\n".join(lines))


def _provenance(df: pd.DataFrame, table: str, cols: tuple[str, ...]) -> pd.Series:
    s = df[cols[0]].astype("string")
    for c in cols[1:]:
        s = s + "/" + df[c].astype("string")
    return table + "#" + s.fillna("?")


class _Problems:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, table: str, problem: str, mask: pd.Series, prov: pd.Series) -> None:
        mask = mask.fillna(True).astype(bool)
        if mask.any():
            self.items.append({"table": table, "problem": problem, "n_rows": int(mask.sum()),
                               "examples": [str(x) for x in prov[mask].head(MAX_EXAMPLES)]})


def _split_values(level: str, patient_id: pd.Series, family_id: pd.Series,
                  admission_id: pd.Series | None) -> pd.Series:
    if level == "family":
        return family_id.astype("string").fillna(patient_id.astype("string"))
    if level in ("admission", "index_row") and admission_id is not None:
        return admission_id.astype("string")
    return patient_id.astype("string")


def tag_tables(tables: dict[str, pd.DataFrame], split_unit: str = rules.DEFAULT_SPLIT_UNIT
               ) -> dict[str, pd.DataFrame]:
    """원본 테이블들 → 꼬리표가 붙은 복사본. 원본은 바꾸지 않는다."""
    level = rules.split_level(split_unit)
    P = _Problems()
    for name in tables:
        if name not in rules.TABLES:
            df = tables[name]
            P.add(name, "규칙표에 이 테이블의 규칙이 없다", pd.Series(True, index=df.index),
                  pd.Series([f"{name}#{i}" for i in df.index], index=df.index))
    for need in ("patients", "admissions"):
        if need not in tables:
            raise TaggingError([{"table": need, "problem": "꼬리표 연결에 필요한 테이블이 없다",
                                 "n_rows": 0, "examples": []}])

    pts = tables["patients"]
    fam = pts.set_index("patient_id")["family_id"] if "family_id" in pts else pd.Series(dtype="string")
    adm = tables["admissions"]
    adm_idx = adm.set_index("admission_id")

    out: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        if name not in rules.TABLES:
            continue
        rule = rules.TABLES[name]
        t = df.copy()
        prov = _provenance(t, name, rule.id_columns)
        for c in rule.id_columns:
            P.add(name, f"식별 열 {c}가 비어 있다", t[c].isna(), prov)

        if rule.entity_via == "patient_id":
            pid = t["patient_id"]
            aid = t["admission_id"] if "admission_id" in t else None
        else:
            aid = t[rule.entity_via]
            P.add(name, "admission_id가 입원 테이블에 없다", ~aid.isin(adm_idx.index), prov)
            pid = aid.map(adm_idx["patient_id"])
        P.add(name, "patient_id가 환자 테이블에 없다", ~pid.isin(pts["patient_id"]), prov)

        if aid is not None:
            t["_admission_id"] = aid
            t["_admit_time"] = aid.map(adm_idx["admit_time"])
            t["_discharge_time"] = aid.map(adm_idx["discharge_time"])

        for c, kind in rule.time_columns.items():
            if kind == "date":
                parsed = pd.to_datetime(t[c], format="%Y-%m-%d", errors="coerce")
                P.add(name, f"날짜 열 {c}를 읽을 수 없다", parsed.isna(), prov)
            else:
                P.add(name, f"시각 열 {c}가 비어 있다", t[c].isna(), prov)

        if name == "admissions":
            P.add(name, "퇴원 시각이 입원 시각보다 이르다", t["discharge_time"] < t["admit_time"], prov)
            t["length_of_stay_h"] = (t["discharge_time"] - t["admit_time"]).dt.total_seconds() / 3600
        if name == "labs":
            P.add(name, "보고 시각이 채취 시각보다 이르다", t["report_time"] < t["collect_time"], prov)

        t["_entity"] = pid.astype("string")
        t["_split_unit"] = _split_values(level, pid, pid.map(fam), aid)
        t["_available_time"] = available_series(t, name, None)
        t["_provenance"] = prov
        out[name] = t

    if P.items:
        raise TaggingError(P.items)
    return out


def _from_avail(t: pd.DataFrame, a: Avail) -> pd.Series:
    if a.kind == "static":
        return pd.Series(T_MIN, index=t.index, dtype="datetime64[ns]")
    if a.kind == "column":
        return t[a.column] + pd.Timedelta(hours=a.offset_h)
    if a.kind == "row_anchor":
        src = {"admit": "_admit_time", "discharge": "_discharge_time"}[a.column]
        return t[src] + pd.Timedelta(hours=a.offset_h)
    if a.kind == "date_end":
        day = pd.to_datetime(t[a.column], format="%Y-%m-%d", errors="coerce")
        return day + pd.Timedelta(hours=24) - pd.Timedelta(minutes=1)
    raise rules.RuleMissing(f"알 수 없는 규칙 종류: {a.kind}")


def available_series(t: pd.DataFrame, table: str, column: str | None) -> pd.Series:
    """꼬리표 붙은 테이블에서 (table, column) 값의 확인 가능 시각."""
    if table == "admissions" and "_admit_time" not in t:
        t = t.assign(_admit_time=t["admit_time"], _discharge_time=t["discharge_time"])
    return _from_avail(t, rules.availability(table, column))
