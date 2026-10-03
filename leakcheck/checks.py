"""Q1~Q7 점검. 사례별 금지 목록이 아니라 세 가지 비교 원리로 판정한다.

1. 시각 비교   (Q1, Q5)  규칙표가 말하는 확인 가능 시각의 상한 ≤ 기준 시점(tₚ)인가
2. 단위 비교   (Q2)      분할 키의 묶음이 선언된 독립 단위 이상인가
3. 범위 비교   (Q3, Q4)  데이터에서 추정한 것의 적합 범위 ⊆ 학습 부분인가 /
                          특징의 행·시간 구간이 결과를 정한 행·결과 창과 겹치는가

Q6는 기록만, Q7은 경고만 한다. 판정은 차단 · 경고 · 통과 · 기록.

설계서만으로 판정하고(기호 시각), 데이터를 주면 같은 원리를 행 단위로 다시 확인해
근거(행 수)를 덧붙인다. 규칙표의 가정과 데이터가 어긋나면 그것도 차단으로 보고한다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from leakcheck import design as dz
from leakcheck import rules, splitting, timeline
from leakcheck.features import UNKNOWN, build_index, make_feature
from leakcheck.tagging import tag_tables
from leakcheck.timeline import INF, fmt_h

BLOCK, WARN, PASS, RECORD = "차단", "경고", "통과", "기록"
QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
Q_NAMES = {"Q1": "가용 시점", "Q2": "독립 단위", "Q3": "적합 범위", "Q4": "결과 출처",
           "Q5": "선택 시점", "Q6": "다중 시도", "Q7": "결과 확인 균질성", "Q1~Q3": "출처 불명"}
TOL = 1e-9


@dataclass
class Finding:
    question: str
    verdict: str
    target: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    design_id: str
    design_type: str
    data_checked: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.verdict == BLOCK for f in self.findings)

    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict in (BLOCK, WARN)]

    def summary(self) -> dict:
        out = {v: 0 for v in (BLOCK, WARN, PASS, RECORD)}
        for f in self.findings:
            out[f.verdict] += 1
        return out

    def to_dict(self) -> dict:
        return {"design_id": self.design_id, "design_type": self.design_type,
                "rules_version": rules.RULES_VERSION, "data_checked": self.data_checked,
                "blocked": self.blocked, "summary": self.summary(),
                "findings": [f.to_dict() for f in self.findings]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_text(self) -> str:
        s = self.summary()
        head = (f"설계 {self.design_id} ({self.design_type}) — 차단 {s[BLOCK]} · 경고 {s[WARN]} · "
                f"통과 {s[PASS]} · 기록 {s[RECORD]}"
                f"{' (데이터 확인 포함)' if self.data_checked else ' (설계서만 확인)'}")
        lines = [head, "결론: " + ("차단 — 설계를 고치기 전에는 모델링을 진행하지 않는다" if self.blocked
                                  else "차단 없음" + (" — 경고는 사람이 확인" if s[WARN] else ""))]
        order = {BLOCK: 0, WARN: 1, RECORD: 2, PASS: 3}
        for f in sorted(self.findings, key=lambda f: (order[f.verdict], f.question)):
            lines.append(f"[{f.verdict}] {f.question} {Q_NAMES.get(f.question, '')} · {f.target}: {f.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 기호 시각

def _scope_bounds(scope: str, fr: timeline.Frame) -> tuple[float, float]:
    """재료 입원의 (입원 시각 상한, 퇴원 시각 상한), tₚ 대비 시간."""
    if scope in ("index_admission", "patient"):
        return fr.hi("admit"), fr.hi("discharge")
    if scope == "prior_admissions":  # 이번 입원 전에 퇴원한 입원
        return fr.hi("admit"), fr.hi("admit")
    if scope in ("next_admission", "patient_history"):
        return INF, INF
    raise ValueError(f"알 수 없는 범위: {scope}")


def avail_upper(spec: dict, fr: timeline.Frame) -> tuple[float, str]:
    """명세가 고르는 행들의 확인 가능 시각 상한(tₚ 대비 시간)과 그 근거."""
    table = spec["source"]
    rule = rules.table_rule(table)
    col = spec.get("column")
    a = rules.availability(table, col)
    scope = spec["scope"]
    adm_hi, dis_hi = _scope_bounds(scope, fr)
    what = f"{table}.{col or rule.value_column or '행'}"
    basis = f"규칙({a.layer}층): {a.basis}"

    if a.kind == "static":
        ub, why = fr.hi("admit"), "정적 정보 → 입원 시각"
    elif a.kind == "row_anchor":
        ub = (adm_hi if a.column == "admit" else dis_hi) + a.offset_h
        why = f"소속 입원의 {'입원' if a.column == 'admit' else '퇴원'} 시각(범위 {scope})"
    else:
        extra = rule.max_after_discharge_h
        ub = dis_hi + extra if extra is not None else INF
        why = (f"범위 {scope}의 행은 퇴원 뒤에도 알려질 수 있음" if extra is None
               else f"범위 {scope}의 퇴원 시각 + {extra:g}h")

    w = spec.get("window")
    end = timeline.parse(w.get("end", "inf")) if w else None
    if end is not None and end.anchor != "inf":
        tc = spec.get("time_column") or rules.default_time_column(table)
        if tc is None:
            raise rules.RuleMissing(f"'{table}'에는 창을 적용할 시각 열이 없다")
        e_hi = fr.hi(end)
        wub, wwhy = INF, f"창 기준 열 {tc}와 확인 가능 시각의 관계가 규칙표에 없음"
        if a.kind == "column":
            if tc == a.column:
                wub, wwhy = e_hi + a.offset_h, f"창 끝 {end} (확인 가능 시각 열 {tc} 기준)"
            elif tc in a.lag_from:
                lag = a.lag_from[tc]
                wub = e_hi + lag if lag is not None else INF
                wwhy = (f"창이 {tc} 기준이라 {a.column}까지 지연 상한 없음" if lag is None
                        else f"창 끝 {end} + {tc}→{a.column} 지연 {lag:g}h")
        elif a.kind == "date_end" and tc == a.column:
            wub, wwhy = e_hi + 24, f"창 끝 {end}의 날짜 끝(23:59)"
        elif a.kind == "row_anchor" and table == "admissions":
            anchor_col = {"admit": "admit_time", "discharge": "discharge_time"}[a.column]
            if tc == anchor_col:
                wub, wwhy = e_hi + a.offset_h, f"창 끝 {end} ({tc} 기준)"
            else:
                wwhy = f"창은 {tc} 기준인데 값은 {anchor_col}에 확정됨"
        elif a.kind == "static":
            wub, wwhy = ub, why
        if wub <= ub:  # 창이 범위보다 좁히거나 같으면 창이 근거
            ub, why = wub, wwhy
    return ub, f"{what}: {why} → 확인 가능 시각 상한 {fmt_h(ub)}. {basis}"


def _span(spec: dict, fr: timeline.Frame, scope: str) -> tuple[float, float] | None:
    """행들이 놓인 시간 구간 (lo, hi], tₚ 대비. 정적 정보는 None."""
    if scope == "patient" or spec.get("source") == "patients":
        return None
    if scope == "index_admission":
        extra = rules.table_rule(spec["source"]).max_after_discharge_h
        lo, hi = fr.lo("admit"), fr.hi("discharge") + (extra if extra is not None else INF)
    elif scope == "prior_admissions":
        lo, hi = -INF, fr.hi("admit")
    elif scope == "next_admission":
        lo, hi = fr.lo("discharge"), INF
    else:
        lo, hi = -INF, INF
    w = spec.get("window")
    if w:
        lo = max(lo, fr.lo(w.get("start", "-inf")))
        hi = min(hi, fr.hi(w.get("end", "inf")))
    return lo, hi


def _overlap(a, b) -> bool:
    return a is not None and b is not None and max(a[0], b[0]) < min(a[1], b[1]) - TOL


def _scopes_overlap(a: str, b: str) -> bool:
    if "patient" in (a, b):
        return a == b
    return a == b or "patient_history" in (a, b)


def _filters_compatible(f1: dict, f2: dict) -> bool:
    for col in set(f1) & set(f2):
        v1, v2 = [str(x) for x in f1[col]], [str(x) for x in f2[col]]
        if col in rules.CODE_COLUMNS:
            if not any(x.startswith(y) or y.startswith(x) for x in v1 for y in v2):
                return False
        elif not set(v1) & set(v2):
            return False
    return True


# ---------------------------------------------------------------- 데이터 단계 준비

@dataclass
class DataCtx:
    tagged: dict
    base: pd.DataFrame      # 포함·제외 적용 전 예측 행
    cohort: pd.DataFrame    # 적용 후


_OPS = {"==": lambda s, v: s == v, "!=": lambda s, v: s != v, ">=": lambda s, v: s >= v,
        ">": lambda s, v: s > v, "<=": lambda s, v: s <= v, "<": lambda s, v: s < v,
        "in": lambda s, v: s.isin(v), "not_in": lambda s, v: ~s.isin(v)}


def _criterion_mask(tagged, spec, rows) -> pd.Series:
    r = make_feature(tagged, spec, rows)
    val = r.frame.set_index("index_id")["value"].reindex(rows["index_id"])
    return np.asarray(_OPS[spec["op"]](val, spec["value"]).fillna(False), dtype=bool)


def prepare_data(d: dict, tables: dict) -> DataCtx:
    tagged = tag_tables(tables, d["split_unit"])
    base = build_index(d, tagged)
    keep = np.ones(len(base), dtype=bool)
    for part, sign in (("inclusion", True), ("exclusion", False)):
        for spec in d["cohort"][part]:
            if not dz.provenance_known(spec):
                continue
            try:
                m = _criterion_mask(tagged, spec, base)
            except rules.RuleMissing:
                continue  # 적용하지 않고 Q5에서 "규칙표에 없어 확인 불가"로 보고한다
            keep &= m if sign else ~m
    return DataCtx(tagged, base, base[keep].reset_index(drop=True))


# ---------------------------------------------------------------- Q1, Q5: 시각 비교

def _time_check(q: str, part: str, specs: list[dict], d: dict, ctx: DataCtx | None, ref: str) -> list[Finding]:
    out = []
    fr_list = dz.frames(d)
    for spec in specs:
        target = f"{part}.{spec['name']}"
        worst, why_w, label = -INF, "", ""
        for fr in fr_list:
            ub, why = avail_upper(spec, fr)
            gap = ub - fr.lo(ref)
            if gap > worst:
                worst, why_w, label = gap, why, fr.label()
        data_note, n_leak = "", 0
        if ctx is not None:
            rows = ctx.base if q == "Q5" else ctx.cohort
            r = make_feature(ctx.tagged, spec, rows)
            leak = r.leaks(rows, ref)
            n_leak = int(leak.sum())
            data_note = f" 데이터: {n_leak:,}/{len(rows):,} 예측 행에서 확인 가능 시각 > {ref}."
        what = "특징" if q == "Q1" else "포함·제외 기준"
        if worst > TOL:
            out.append(Finding(q, BLOCK, target,
                               f"{what}의 확인 가능 시각이 기준 시점({ref}) 뒤일 수 있다 (tₚ = {label}). "
                               f"{why_w}{data_note}"))
        elif n_leak > 0:
            out.append(Finding(q, BLOCK, target,
                               f"규칙표로는 {ref} 이전이지만 데이터에서 늦게 알려진 행이 있다 "
                               f"(규칙표 가정과 데이터 불일치). {why_w}{data_note}"))
    return out


# ---------------------------------------------------------------- Q2: 단위 비교

def _find_key(obj, key: str, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k == key:
                yield p, v
            yield from _find_key(v, key, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_key(v, key, f"{path}[{i}]")


def check_q2(d: dict, ctx: DataCtx | None) -> list[Finding]:
    out = []
    L = rules.SPLIT_LEVELS
    unit = rules.split_level(d["split_unit"])
    if L[unit] < L[rules.ENTITY_LEVEL]:
        out.append(Finding("Q2", BLOCK, "split_unit",
                           f"독립 단위 '{unit}'이 개체 단위(환자)보다 작다. 같은 환자의 행은 함께 움직여야 한다."))
    split = d["split"]
    level = splitting.effective_key_level(split)
    shown = split["key"]
    if split.get("method") in splitting.ROW_LEVEL_METHODS:
        shown += f" (방법 {split['method']}: 행 단위)"
    keys = [("split.key", level, shown)]
    for path, v in _find_key(d.get("model", {}), "cv_key", "model"):
        keys.append((path, rules.split_level(v), v))
    for target, lv, shown in keys:
        if L[lv] < L[unit]:
            note = ""
            if ctx is not None and target == "split.key":
                labels = splitting.assign(ctx.cohort, split)
                n, tot = splitting.crossing(ctx.cohort, labels, unit)
                note = f" 데이터: {unit} {n:,}/{tot:,}개가 둘 이상의 배정에 나뉨."
            out.append(Finding("Q2", BLOCK, target,
                               f"분할 묶음 {shown}({lv})이 선언된 독립 단위 {unit}보다 작아 같은 {unit}의 "
                               f"행이 학습·평가에 함께 들어갈 수 있다.{note}"))
    if ctx is not None and L[level] >= L[unit]:
        labels = splitting.assign(ctx.cohort, split)
        for lv in [x for x in L if L[x] <= L[unit] and L[x] >= L["admission"]]:
            n, tot = splitting.crossing(ctx.cohort, labels, lv)
            if n:
                out.append(Finding("Q2", BLOCK, "split.key",
                                   f"데이터에서 {lv} {n:,}/{tot:,}개가 학습·평가에 나뉨 (선언 단위 {unit} 이하)."))
        for lv in [x for x in L if L[x] > L[unit]]:
            n, tot = splitting.crossing(ctx.cohort, labels, lv)
            if n:
                out.append(Finding("Q2", WARN, "split_unit",
                                   f"독립 단위를 {unit}로 선언했는데 데이터에서 같은 {lv} {n:,}/{tot:,}개가 "
                                   f"학습·평가에 나뉜다. {lv} 구성원이 위험을 공유하면 서로 독립이 아니므로 "
                                   f"단위를 {lv}로 할지 사람이 확인."))
    return out


# ---------------------------------------------------------------- Q3: 적합 범위

def check_q3(d: dict) -> list[Finding]:
    out = []
    ok = rules.FIT_SCOPES_OK
    for i, step in enumerate(d["preprocessing"]):
        t = f"preprocessing.{step['name']}"
        if step.get("stateless"):
            continue
        fs = step.get("fit_scope")
        if fs is None:
            out.append(Finding("Q3", BLOCK, t, f"데이터에서 값을 추정하는 단계({step['kind']})인데 적합 범위가 "
                                               f"적혀 있지 않아 학습 부분에서만 추정했는지 확인할 수 없다."))
        elif fs not in ok:
            out.append(Finding("Q3", BLOCK, t, f"{step['kind']} 단계를 '{fs}' 범위에서 적합한다. 평가 부분의 "
                                               f"정보가 학습에 들어간다. 적합 범위는 train(또는 train_fold)이어야 한다."))
    rest = {k: v for k, v in d.items() if k != "preprocessing"}
    for path, fs in _find_key(rest, "fit_scope"):
        if fs not in ok:
            out.append(Finding("Q3", BLOCK, path.rsplit(".fit_scope", 1)[0],
                               f"데이터에서 추정하는 값을 '{fs}' 범위에서 적합한다. train(또는 train_fold)이어야 한다."))
    return out


# ---------------------------------------------------------------- Q4: 결과 출처

def check_q4(d: dict, feats: list[dict]) -> list[Finding]:
    out = []
    o = d["outcome"]
    proxies = rules.OUTCOME_PROXIES.get(o["definition"], []) + o["proxies"]
    fr_list = dz.frames(d)
    for f in feats:
        t = f"features.{f['name']}"
        same_rows = (f["source"] == o["source"] and _filters_compatible(f["filter"], o["filter"])
                     and _scopes_overlap(f["scope"], o["scope"]))
        hit = None
        for fr in fr_list:
            fs, os_ = _span(f, fr, f["scope"]), _span(o, fr, o["scope"])
            if same_rows and _overlap(fs, os_):
                hit = (f"결과를 정한 행({o['source']}, 범위 {o['scope']}, 결과 창 "
                       f"({o['window'].get('start', '-inf')}, {o['window'].get('end', 'inf')}])과 같은 행을 "
                       f"특징이 쓴다 (tₚ = {fr.label()}).")
                break
            if f.get("window") and _overlap(fs, os_):
                hit = (f"특징의 창 ({f['window'].get('start', '-inf')}, {f['window'].get('end', 'inf')}]이 "
                       f"결과 창과 겹친다 (결과 창 재사용, tₚ = {fr.label()}).")
                break
        if hit:
            out.append(Finding("Q4", BLOCK, t, hit))
            continue
        for p in proxies:
            if (f["source"] == p["source"] and _filters_compatible(f["filter"], p.get("filter", {}))
                    and _scopes_overlap(f["scope"], o["scope"])):
                flt = ", ".join(f"{k}∈{v}" for k, v in p.get("filter", {}).items())
                out.append(Finding("Q4", WARN, t,
                                   f"결과({o['name']})의 대리 변수 목록과 일치: {p['source']} {flt} "
                                   f"({p.get('basis', '설계서 목록')}). 결과를 정하는 과정이 남긴 기록일 수 있어 "
                                   f"사람이 확인."))
                break
    return out


# ---------------------------------------------------------------- Q6, Q7

def check_q6(d: dict) -> list[Finding]:
    a = d.get("attempts")
    if not a:
        return [Finding("Q6", RECORD, "attempts", "시도 횟수가 적혀 있지 않다. 설계·모델을 몇 번 시도했는지 기록할 것.")]
    return [Finding("Q6", RECORD, "attempts",
                    f"설계 {a.get('n_designs_tried', '?')}개, 모델 {a.get('n_models_tried', '?')}개 시도, "
                    f"보정: {a.get('correction', '미기재')}.")]


def _restricted_levels(d: dict, column: str) -> bool:
    for spec in d["cohort"]["inclusion"]:
        if spec.get("column") == column and (
                spec["op"] == "==" or (spec["op"] == "in" and len(spec["value"]) == 1)):
            return True
    return False


def check_q7(d: dict, ctx: DataCtx | None) -> list[Finding]:
    o = d["outcome"]
    rule = rules.table_rule(o["source"])
    if not rule.measurement:
        return []
    handled = set(o["ascertainment"].get("by_stratum", []))
    out = []
    if ctx is None:
        for col in dict.fromkeys([c for _, c in rules.ASCERTAINMENT_STRATA]):
            if col in handled or _restricted_levels(d, col):
                continue
            out.append(Finding("Q7", WARN, f"outcome.{o['name']}",
                               f"결과가 측정({o['source']})으로 정해지는데 측정 강도가 다를 수 있는 층 '{col}'을 "
                               f"함께 두고 층별로 다루지 않는다. 데이터 없이 측정 빈도 비를 확인할 수 없어 사람이 확인."))
        return out
    cohort = ctx.cohort
    adm = ctx.tagged["admissions"].set_index("admission_id")
    pts = ctx.tagged["patients"].set_index("patient_id")
    from leakcheck.features import _apply_filter
    meas = _apply_filter(ctx.tagged[o["source"]], o["filter"])
    adms = cohort.drop_duplicates("admission_id")[["admission_id", "patient_id"]]
    n = meas.groupby("_admission_id").size().reindex(adms["admission_id"], fill_value=0).to_numpy()
    los_d = adm.loc[adms["admission_id"], "length_of_stay_h"].to_numpy() / 24
    rate = pd.Series(n / los_d, index=adms.index)
    cols = list(dict.fromkeys([c for _, c in rules.ASCERTAINMENT_CANDIDATES] + d["cohort"]["subgroups"]))
    for col in cols:
        if col in adm.columns:
            g = adm.loc[adms["admission_id"], col].to_numpy()
        elif col in pts.columns:
            g = pts.loc[adms["patient_id"], col].to_numpy()
        else:
            continue
        by = rate.groupby(g).mean()
        if len(by) < 2:
            continue
        ratio = by.max() / by.min() if by.min() > 0 else INF
        if ratio >= rules.Q7_RATIO_THRESHOLD and col not in handled:
            detail = ", ".join(f"{k} {v:.2f}" for k, v in by.items())
            out.append(Finding("Q7", WARN, f"outcome.{o['name']}",
                               f"결과 측정 빈도(회/일)가 '{col}' 층마다 다르다: {detail} (비 {ratio:.1f} ≥ "
                               f"{rules.Q7_RATIO_THRESHOLD:g}). 자주 측정하는 층에서 결과가 더 많이 확인된다. "
                               f"층별로 다루는지 사람이 확인."))
    return out


# ---------------------------------------------------------------- 실행

def run_checks(design: dict, tables: dict | None = None) -> Report:
    d = dz.normalize(design)
    ctx = prepare_data(d, tables) if tables is not None else None
    findings: list[Finding] = []

    feats, crit = [], {"inclusion": [], "exclusion": []}
    for f in d["features"]:
        if dz.provenance_known(f):
            feats.append(f)
        else:
            findings.append(Finding("Q1~Q3", WARN, f"features.{f['name']}",
                                    f"{UNKNOWN}: 정해진 특징 함수(leakcheck.features)로 만들지 않았거나 원본 테이블이 "
                                    f"없다. Q1~Q3 확인 불가."))
    for part in ("inclusion", "exclusion"):
        for c in d["cohort"][part]:
            if dz.provenance_known(c):
                crit[part].append(c)
            else:
                findings.append(Finding("Q5", WARN, f"cohort.{part}.{c['name']}",
                                        f"{UNKNOWN}: 기준의 원본 테이블이 없어 선택 시점을 확인할 수 없다."))

    def guarded(question, target, fn):
        try:
            return fn()
        except rules.RuleMissing as e:
            return [Finding(question, WARN, target, f"규칙표에 없어 확인 불가: {e}. 규칙을 추가해야 한다.")]

    ref = d["cohort"]["index_time"]
    q1 = []
    for f in feats:
        q1 += guarded("Q1~Q3", f"features.{f['name']}", lambda f=f: _time_check("Q1", "features", [f], d, ctx, "tp"))
    findings += q1
    findings += check_q2(d, ctx)
    findings += check_q3(d)
    findings += guarded("Q4", f"outcome.{d['outcome']['name']}", lambda: check_q4(d, feats))
    for part in ("inclusion", "exclusion"):
        for c in crit[part]:
            findings += guarded("Q5", f"cohort.{part}.{c['name']}",
                                lambda c=c, part=part: _time_check("Q5", f"cohort.{part}", [c], d, ctx, ref))
    findings += check_q6(d)
    findings += guarded("Q7", f"outcome.{d['outcome']['name']}", lambda: check_q7(d, ctx))

    seen = {f.question for f in findings}
    checked = {
        "Q1": f"특징 {len(feats)}개의 확인 가능 시각 ≤ tₚ",
        "Q2": f"분할 키 {d['split']['key']} ≥ 독립 단위 {d['split_unit']}",
        "Q3": f"전처리 {len(d['preprocessing'])}단계의 적합 범위 = 학습 부분",
        "Q4": "특징이 결과 행·결과 창·대리 변수와 겹치지 않음",
        "Q5": f"포함·제외 기준 {len(crit['inclusion']) + len(crit['exclusion'])}개의 확인 가능 시각 ≤ {ref}",
        "Q7": "결과 확인 강도가 층마다 같음 (또는 층별로 다룸, 또는 측정 의존 결과 아님)",
    }
    for q, txt in checked.items():
        if q not in seen:
            findings.append(Finding(q, PASS, "설계 전체", txt))
    return Report(d["design_id"], d["design_type"], ctx is not None, findings)
