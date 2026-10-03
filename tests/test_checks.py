import copy
import csv
import re
from pathlib import Path

import pytest

from leakcheck import rules
from leakcheck.checks import BLOCK, PASS, RECORD, WARN, run_checks
from leakcheck.design import DesignError
from tests.design_mutations import TRICKY_DYNAMIC, clean, inject, tricky, tricky_all

ROOT = Path(__file__).resolve().parent.parent
KINDS = {"동적": ["동적"], "고정": ["고정"], "둘다": ["동적", "고정"]}


def _public_cases():
    with open(ROOT / "designs" / "error_catalog_public.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [pytest.param(r, k, id=f"{r['id']}-{k}") for r in rows for k in KINDS[r["design_types"]]]


def _problems(report):
    return [(f.question, f.verdict) for f in report.problems()]


# --- 깨끗한 설계 ---

@pytest.mark.parametrize("kind", ["동적", "고정"])
def test_clean_design_passes(kind):
    r = run_checks(clean(kind))
    assert r.problems() == []
    assert {f.question for f in r.findings if f.verdict == PASS} == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q7"}
    assert [f.question for f in r.findings if f.verdict == RECORD] == ["Q6"]


@pytest.mark.parametrize("kind", ["동적", "고정"])
def test_clean_design_with_data_only_family_warning(kind, small_tables):
    """깨끗한 설계(독립 단위 = 환자)는 데이터 단계에서 가족 경고만 받는다."""
    r = run_checks(clean(kind), small_tables)
    assert not r.blocked
    assert [(f.question, f.verdict, f.target) for f in r.problems()] == [("Q2", WARN, "split_unit")]


def test_family_split_has_no_family_warning(small_tables):
    d = clean("동적")
    d["split_unit"], d["split"]["key"], d["model"]["tuning"]["cv_key"] = "family", "family_id", "family_id"
    assert run_checks(d, small_tables).problems() == []


# --- 공개 12개 사례 ---

@pytest.mark.parametrize("case,kind", _public_cases())
def test_public_case_gets_expected_verdict(case, kind):
    r = run_checks(inject(case["id"], kind))
    assert (case["question"], case["expected_verdict"]) in _problems(r), r.to_text()


@pytest.mark.parametrize("case,kind", _public_cases())
def test_public_case_with_data(case, kind, small_tables):
    r = run_checks(inject(case["id"], kind), small_tables)
    assert (case["question"], case["expected_verdict"]) in _problems(r), r.to_text()


# --- 수상해 보이지만 정당한 설계 ---

@pytest.mark.parametrize("name", list(TRICKY_DYNAMIC))
def test_suspicious_but_legitimate_feature_passes(name):
    assert run_checks(tricky(name)).problems() == []


def test_suspicious_but_legitimate_design_passes_with_data(small_tables):
    r = run_checks(tricky_all(), small_tables)
    assert [(f.question, f.verdict, f.target) for f in r.problems()] == [("Q2", WARN, "split_unit")]
    assert "데이터" not in " ".join(f.reason for f in r.findings if f.question == "Q1")


# --- 핵심 시연: 같은 규칙, 다른 tₚ ---

DX_FEATURE = {"name": "dx_ckd_this_admission", "source": "diagnoses", "scope": "index_admission",
              "filter": {"icd_code": ["N18"]}, "agg": "any"}


def _with_dx(kind):
    d = clean(kind)
    d["features"] = [f for f in d["features"] if f["name"] != "dx_ckd"] + [copy.deepcopy(DX_FEATURE)]
    return d


def test_core_demo_same_rule_blocks_dynamic_passes_fixed(small_tables):
    dyn = run_checks(_with_dx("동적"))
    fix = run_checks(_with_dx("고정"))
    assert ("Q1", BLOCK, "features.dx_ckd_this_admission") in [(f.question, f.verdict, f.target) for f in dyn.findings]
    assert fix.problems() == []
    # 데이터에서도 같은 결론: 동적은 모든 예측 행이 퇴원 전, 고정은 tₚ = 퇴원
    dyn_d = run_checks(_with_dx("동적"), small_tables)
    fix_d = run_checks(_with_dx("고정"), small_tables)
    assert dyn_d.blocked and not fix_d.blocked


def test_coding_delay_rule_flips_fixed_design(monkeypatch):
    """규칙표 한 줄(코딩 지연)을 바꾸면 고정 시점 설계에서도 진단 코드가 차단된다."""
    delayed = rules.Avail("row_anchor", "discharge", 2, "퇴원 후 코딩 지연 72h (가정)", offset_h=72.0)
    dx = rules.TABLES["diagnoses"]
    monkeypatch.setitem(rules.TABLES, "diagnoses",
                        rules.TableRule(**{**dx.__dict__, "row_available": delayed, "max_after_discharge_h": 72.0}))
    assert ("Q1", BLOCK) in _problems(run_checks(_with_dx("고정")))


# --- 사용자 요청 반영 ---

def test_q4_same_table_alone_is_not_blocked():
    """동적 AKI: 특징도 결과도 크레아티닌(labs)이지만 행·시간 구간이 겹치지 않으면 통과."""
    d = clean("동적")
    assert d["outcome"]["source"] == "labs"
    assert any(f.get("source") == "labs" and f.get("filter", {}).get("test") == ["creatinine"] for f in d["features"])
    r = run_checks(d)
    assert [f for f in r.findings if f.question == "Q4"][0].verdict == PASS


def test_q4_window_overlapping_outcome_window_blocks():
    d = clean("동적")
    d["features"].append({"name": "cr_next_24h", "source": "labs", "filter": {"test": ["creatinine"]},
                          "time_column": "report_time", "window": {"start": "tp", "end": "tp+24h"},
                          "agg": "max"})
    found = [(f.question, f.verdict) for f in run_checks(d).problems() if "cr_next_24h" in f.target]
    assert ("Q4", BLOCK) in found and ("Q1", BLOCK) in found


def test_q4_outcome_window_reuse_from_other_table_blocks():
    d = clean("동적")
    d["features"].append({"name": "orders_in_outcome_window", "source": "orders",
                          "filter": {"order_type": ["chest_xray"]}, "time_column": "order_time",
                          "window": {"start": "tp", "end": "tp+48h"}, "agg": "count"})
    assert ("Q4", BLOCK) in _problems(run_checks(d))


def test_q2_family_crossing_is_data_warning(small_tables):
    r = run_checks(clean("고정"), small_tables)
    w = [f for f in r.findings if f.question == "Q2" and f.verdict == WARN]
    assert len(w) == 1 and "family" in w[0].reason and re.search(r"\d+/\d+", w[0].reason)
    assert not any(f.question == "Q2" and f.verdict == WARN for f in run_checks(clean("고정")).findings)


def test_unknown_provenance_feature_reports_q1_q3_unverifiable():
    d = clean("동적")
    d["features"].append({"name": "risk_score", "description": "노트북에서 계산"})
    d["features"].append({"name": "cr_slope", "source": "labs", "made_by": "pandas 직접 계산"})
    r = run_checks(d)
    unk = [f for f in r.findings if f.question == "Q1~Q3"]
    assert {f.target for f in unk} == {"features.risk_score", "features.cr_slope"}
    assert all(f.verdict == WARN and "Q1~Q3 확인 불가" in f.reason for f in unk)


def test_q7_threshold_is_fixed_at_two():
    assert rules.Q7_RATIO_THRESHOLD == 2.0


def test_q7_restricted_cohort_passes():
    d = clean("동적")
    d["outcome"].pop("ascertainment")
    d["cohort"]["inclusion"].append({"name": "icu_only", "source": "admissions", "column": "unit",
                                     "agg": "value", "op": "==", "value": "ICU"})
    assert not any(f.question == "Q7" and f.verdict == WARN for f in run_checks(d).findings)


def test_q3_missing_fit_scope_blocks_and_stateless_passes():
    d = clean("고정")
    d["preprocessing"].append({"name": "log_cr", "kind": "transform", "stateless": True})
    assert run_checks(d).problems() == []
    d["preprocessing"].append({"name": "pca", "kind": "reduce"})
    assert ("Q3", BLOCK) in _problems(run_checks(d))


def test_q3_fit_scope_anywhere_in_design_is_checked():
    d = clean("고정")
    d["model"]["tuning"]["fit_scope"] = "train+test"
    assert ("Q3", BLOCK) in _problems(run_checks(d))


def test_row_level_split_method_blocks_even_with_patient_key():
    d = clean("동적")
    d["split"]["method"] = "random_rows"
    assert ("Q2", BLOCK) in _problems(run_checks(d))


def test_unit_finer_than_entity_blocks():
    d = clean("고정")
    d["split_unit"], d["split"]["key"] = "admission", "admission_id"
    assert ("Q2", BLOCK) in _problems(run_checks(d))


def test_unknown_source_is_reported_not_guessed():
    d = clean("동적")
    d["features"].append({"name": "hr_last", "source": "vitals", "agg": "last"})
    f = [f for f in run_checks(d).findings if f.target == "features.hr_last"]
    assert f and f[0].verdict == WARN and "규칙표에 없어 확인 불가" in f[0].reason


def test_unknown_source_or_column_with_data_is_reported(small_tables):
    d = clean("동적")
    d["features"].append({"name": "cr_flag", "source": "labs", "column": "flag", "agg": "last"})
    d["cohort"]["exclusion"].append({"name": "on_pressors", "source": "vitals", "agg": "any",
                                     "op": "==", "value": True})
    r = run_checks(d, small_tables)
    got = {f.target: (f.question, f.verdict) for f in r.findings if "확인 불가" in f.reason}
    assert got == {"features.cr_flag": ("Q1~Q3", WARN), "cohort.exclusion.on_pressors": ("Q5", WARN)}


def test_invalid_design_raises():
    d = clean("동적")
    d["features"][0]["window"] = {"end": "tomorrow"}
    with pytest.raises(DesignError):
        run_checks(d)
    d = clean("고정")
    del d["split"]
    with pytest.raises(DesignError):
        run_checks(d)


def test_output_is_structured_json():
    import json
    r = run_checks(inject("E02", "고정"))
    out = json.loads(r.to_json())
    assert out["blocked"] is True
    assert all(set(f) == {"question", "verdict", "target", "reason"} for f in out["findings"])


def test_check_code_knows_no_case_ids():
    """점검은 사례 목록이 아니라 원리로 구현한다: 점검·스킬 코드에 사례 ID가 없어야 한다."""
    pat = re.compile(r"\bE\d{2}\b")
    for p in list((ROOT / "leakcheck").rglob("*.py")) + list((ROOT / "skill_src").rglob("*")):
        if p.is_file():
            assert not pat.search(p.read_text(encoding="utf-8")), p
