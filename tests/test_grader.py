"""5단계 채점기 시험. 진짜 정답표는 쓰지 않는다 (가짜 정답표 + 가짜 보고서).

가짜 변형은 기본 설계서에 공개 사례 패치를 적용해 만든다 (designs/inject.py와 같은 방식).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from designs import inject as inj
from experiment import grader as gr
from experiment import prompt

ROOT = Path(__file__).resolve().parent.parent
BASES = inj.load_bases()


def variant(t: str, case_ids: list[str] = (), legit: str | None = None) -> tuple[dict, dict]:
    """(가짜 변형 설계서, 정답표 항목) — inject.build와 같은 구조."""
    cases = {c["id"]: c for c in inj.public_cases()}
    d = BASES[t]
    entry = {"design_type": t, "defects": [], "legit_changes": [], "sha256": "-"}
    for cid in case_ids:
        c = cases[cid]
        d = inj.apply_ops(d, c["ops"])
        entry["defects"].append({k: c[k] for k in ("id", "name", "question", "expected_verdict",
                                                   "holdout", "adjustment", "targets")})
    if legit:
        ops = inj.LEGIT_CHANGES[legit]["ops"]
        d = inj.apply_ops(d, ops)
        entry["legit_changes"].append({"name": legit, "targets": inj.targets_of(ops), "note": "결함 아님"})
    return d, entry


def as_holdout(entry: dict, adjusted: bool = False) -> dict:
    e = copy.deepcopy(entry)
    for d in e["defects"]:
        d["holdout"], d["adjustment"] = True, ("가짜 조정" if adjusted else None)
    return e


def report(items: list[dict] | None, prose: str = "설계를 검토했다.") -> str:
    if items is None:
        return prose
    return prose + "\n\n```findings\n" + json.dumps(items, ensure_ascii=False, indent=1) + "\n```\n"


def grade(design, entry, items, prose="설계를 검토했다.", rid="r1"):
    return gr.grade_report({"report_id": rid, "variant": "v.json", "text": report(items, prose)}, design, entry)


# ---------------------------------------------------------------- 목록 추출

def test_parse_formats():
    assert gr.parse_findings("목록 없이 서술만") == ([], "목록 없음", 0)
    assert gr.parse_findings("```findings\n[{bad json\n```")[1] == "JSON 오류"
    assert gr.parse_findings("```findings\n{\"target\": \"x\"}\n```")[1] == "목록 아님"
    items, err, bad = gr.parse_findings(report([
        {"target": "features.age", "problem": "a"},
        {"target": ["split.key", "model.tuning.cv_key"], "problem": "b"},
        {"problem": "대상 없음"}, "문자열"]))
    assert err is None and bad == 2
    assert [x["target"] for x in items] == ["features.age", "split.key", "model.tuning.cv_key"]
    assert gr.parse_findings(report([]))[:2] == ([], None)


def test_only_last_block_and_prose_ignored():
    d, e = variant("dynamic", ["E02"])
    text = ("본문에서 features.cr_last가 채취 시각 기준이라 미래 정보라고 썼다.\n"
            "```findings\n[{\"target\": \"features.age\", \"problem\": \"초안\"}]\n```\n"
            "```findings\n[]\n```")
    g = gr.grade_report({"report_id": "r", "variant": "v.json", "text": text}, d, e)
    assert g.defects[0]["primary"] is False and g.false_alarms == []


# ---------------------------------------------------------------- 문제 종류

@pytest.mark.parametrize("text,q", [
    ("채취 시각으로 걸러서 tₚ 이후에 보고된 값이 들어간다", "Q1"),
    ("uses information not yet available at prediction time", "Q1"),
    ("같은 환자의 입원이 학습과 평가에 나뉘어 들어간다", "Q2"),
    ("rows from the same patient end up in both train and test sets", "Q2"),
    ("중앙값 대치를 전체 데이터로 적합한다", "Q3"),
    ("the selector is fit on the full dataset before splitting", "Q3"),
    ("결과를 정한 행과 같은 행을 쓴다", "Q4"),
    ("this order is a proxy for the outcome (target leakage)", "Q4"),
    ("불멸 시간 편향이 생긴다", "Q5"),
    ("ICU와 병동의 측정 빈도가 달라 결과 확인 강도가 다르다", "Q7"),
    ("surveillance bias: ICU patients are tested more often", "Q7"),
    ("[차단] Q3 적합 범위", "Q3"),
])
def test_keywords(text, q):
    k = gr.classify(text)
    assert k.status == "kinds" and q in k.qs


@pytest.mark.parametrize("text,q", [   # 5단계 시험에서 종류 불명으로 떨어졌던 자연스러운 표현 (사전 보강)
    ("이번 입원의 진단명은 퇴원 후에 코딩되어 예측 시점에 사용할 수 없습니다.", "Q1"),
    ("가족 구성원이 학습과 검증에 나뉘어 들어가 독립성이 깨집니다.", "Q2"),
    ("행 단위로 나누면 한 사람의 데이터가 양쪽에 섞여 성능이 과대평가됩니다.", "Q2"),
    ("변수 선택을 전체 표본에서 하면 시험 세트 정보를 미리 본 것입니다.", "Q3"),
    ("다음 입원의 정보를 쓰는데, 다음 입원 자체가 예측하려는 결과입니다.", "Q4"),
])
def test_keywords_natural_expressions(text, q):
    k = gr.classify(text)
    assert k.status == "kinds" and q in k.qs


def test_kind_special_cases():
    assert gr.classify("이 특징은 좀 이상하다").status == "unknown"
    assert gr.classify("이 값은 적합하지 않아 보인다").status == "unknown"   # '적합'(알맞음)은 Q3가 아님
    assert gr.classify("시도 횟수를 기록해야 한다 (multiple testing)").status == "q6"
    assert gr.classify("규칙표에 없어 확인 불가: Q1 가용 시점").status == "unable"   # 점검 불가가 먼저
    assert gr.classify("출처 불명: Q1~Q3 확인 불가").status == "unable"
    assert gr.classify("Q1~Q3 위반").qs == {"Q1", "Q2", "Q3"}
    # 포함·제외 기준의 시점 문제는 선택 시점(Q5)이기도 하다
    assert gr.classify("퇴원 때 확정되는 미래 정보", "cohort.inclusion.los_7d").qs >= {"Q1", "Q5"}
    assert "Q5" not in gr.classify("퇴원 때 확정되는 미래 정보", "features.los_h").qs


# ---------------------------------------------------------------- 대상 항목

@pytest.mark.parametrize("target,item", [
    ("features.cr_last", "features.cr_last"),
    ("features[name=cr_last].time_column", "features.cr_last"),
    ("features:cr_last", "features.cr_last"),
    ("`features.cr_last` 특징의 시간 기준", "features.cr_last"),
    ("cr_last", "features.cr_last"),
    ("preprocessing[name=median_impute].fit_scope", "preprocessing.median_impute"),
    ("cohort.exclusion.aki_known_by_tp", "cohort.exclusion.aki_known_by_tp"),
    ("split.key", "split"), ("split_unit", "split"), ("model.tuning.cv_key", "split"),
    ("outcome.aki_48h", "outcome"), ("outcome.window", "outcome.window"),
    ("outcome.ascertainment", "outcome.ascertainment"),
    ("cohort.subgroups (unit)", "cohort.subgroups"),
    ("model.family", "model"),
])
def test_resolve(target, item):
    assert gr.resolve(target, BASES["dynamic"]) == item


@pytest.mark.parametrize("target", ["features", "features.nonexistent", "설계 전체", "features.age, features.sex", "cohort"])
def test_resolve_unresolved(target):
    assert gr.resolve(target, BASES["dynamic"]) is None


# ---------------------------------------------------------------- 탐지: 주 분석(항목) vs 보조 분석(항목+종류)

def test_primary_and_secondary_detection():
    d, e = variant("dynamic", ["E02"])                      # Q1, features.cr_last
    g = grade(d, e, [{"target": "features.cr_last", "problem": "채취 시각 기준이라 tₚ 이후 보고 값이 섞인다"}])
    assert g.defects[0]["secondary"] and g.defects[0]["primary"] and g.false_alarms == []


def test_right_item_wrong_kind_is_primary_only():
    d, e = variant("dynamic", ["E02"])
    g = grade(d, e, [{"target": "features.cr_last", "problem": "결측 대치를 전체 데이터로 적합한다"}])
    assert (g.defects[0]["secondary"], g.defects[0]["primary"]) == (False, True)
    assert g.defects[0]["extra_kinds"] == ["Q3"]


def test_unknown_kind_is_primary_only_and_counted():
    d, e = variant("dynamic", ["E02"])
    g = grade(d, e, [{"target": "features.cr_last", "problem": "이상할 수도 있다"}])
    assert (g.defects[0]["secondary"], g.defects[0]["primary"]) == (False, True)
    assert g.n_unknown_kind == 1 and g.false_alarms == []


def test_vague_prose_without_list_is_miss():
    d, e = variant("dynamic", ["E02"])
    g = grade(d, e, None, prose="cr_last는 채취 시각 기준이라 tₚ 이후 값이 섞일 수 있다.")
    assert g.format_error == "목록 없음"
    assert not g.defects[0]["primary"] and g.false_alarms == []


def test_warning_level_counts():
    d, e = variant("dynamic", ["E13"])                      # 기대 판정 경고 (Q4)
    g = grade(d, e, [{"target": "features.renal_ordered",
                      "problem": "투석 오더는 AKI 결과의 대리 변수일 수 있어 사람이 확인해야 한다"}])
    assert g.defects[0]["secondary"]


# ---------------------------------------------------------------- 추가 판정 vs 오경보

def test_extra_kind_on_defect_item_is_not_false_alarm():
    d, e = variant("fixed", ["E12"])                        # Q4, features.ward_type
    g = grade(d, e, [{"target": "features.ward_type",
                      "problem": "다음 입원 정보라 퇴원 시점에는 아직 알 수 없는 미래 정보이고, 결과를 정한 행과 같은 행이다"}])
    assert g.defects[0]["secondary"] and g.defects[0]["extra_kinds"] == ["Q1"]
    assert g.false_alarms == []


def test_two_entries_on_defect_item():
    d, e = variant("fixed", ["E12"])
    g = grade(d, e, [{"target": "features.ward_type", "problem": "미래 정보"},
                     {"target": "features.ward_type", "problem": "결과 정의에 쓰인 행"}])
    assert g.defects[0]["secondary"] and g.false_alarms == []


def test_false_alarm_on_undefected_item_deduplicated():
    d, e = variant("dynamic", ["E02"])
    g = grade(d, e, [{"target": "features.cr_last", "problem": "미래 정보"},
                     {"target": "features.age", "problem": "나이는 tₚ 이후 정보"},
                     {"target": "features[name=age]", "problem": "다시 지적"},
                     {"target": "preprocessing.standardize", "problem": "이상하다"}])
    assert g.defects[0]["secondary"]
    assert g.false_alarms == ["features.age", "preprocessing.standardize"]


def test_clean_variant_any_flag_is_false_alarm():
    d, e = variant("fixed")
    g = grade(d, e, [{"target": "features.dx_ckd", "problem": "진단 코드는 퇴원 후에야 확정되는 미래 정보"}])
    assert g.defects == [] and g.false_alarms == ["features.dx_ckd"]
    assert grade(d, e, []).false_alarms == []


def test_legit_change_flag_is_false_alarm():
    d, e = variant("dynamic", legit="prior_admission_dx")
    g = grade(d, e, [{"target": "features.prior_dx_renal_cardiac", "problem": "진단 코드는 퇴원 시각에야 알려진다"}])
    assert g.false_alarms == ["features.prior_dx_renal_cardiac"]


def test_not_graded_items_and_q6():
    d, e = variant("dynamic")
    g = grade(d, e, [{"target": "model.family", "problem": "로지스틱 회귀는 단순하다"},
                     {"target": "attempts", "problem": "다중 시도 보정이 없다"},
                     {"target": "설계 전체", "problem": "합성 데이터라 일반화가 어렵다"}])
    assert g.false_alarms == [] and g.n_q6 == 1 and g.n_unresolved == 1


# ---------------------------------------------------------------- 점검 불가

def test_unable_on_defect_is_miss_not_false_alarm():
    d, e = variant("dynamic", ["E02"])
    g = grade(d, e, [{"target": "features.cr_last", "problem": "규칙표에 없어 확인 불가"},
                     {"target": "features.age", "problem": "출처 불명: Q1~Q3 확인 불가"}])
    assert not g.defects[0]["primary"] and g.false_alarms == [] and g.n_unable == 2


# ---------------------------------------------------------------- 동의어 묶음

@pytest.mark.parametrize("target", ["split.key", "split_unit", "split", "model.tuning.cv_key"])
def test_split_group(target):
    d, e = variant("dynamic", ["E07"])                      # Q2, split.key = patient_id
    g = grade(d, e, [{"target": target, "problem": "같은 가족이 학습과 평가에 나뉜다"}])
    assert g.defects[0]["secondary"] and g.false_alarms == []


@pytest.mark.parametrize("target,hit", [("outcome", True), ("outcome.aki_48h", True),
                                        ("outcome.ascertainment", True), ("outcome.window", False),
                                        ("cohort.subgroups", True)])
def test_outcome_group(target, hit):
    d, e = variant("dynamic", ["E17"])                      # Q7, outcome.ascertainment 삭제
    g = grade(d, e, [{"target": target, "problem": "ICU와 병동의 측정 빈도가 달라 결과 확인 강도가 다르다"}])
    assert g.defects[0]["secondary"] is hit
    assert (g.false_alarms == []) is hit


def test_subgroups_only_for_ascertainment():
    """cohort.subgroups는 정답이 outcome.ascertainment일 때만 같은 항목. 그 밖에는 오경보."""
    d = BASES["dynamic"]
    assert gr.matches("cohort.subgroups", gr.key_target("outcome.ascertainment", d))
    assert gr.matches("cohort.subgroups", "outcome.ascertainment.by_stratum")
    for key in ("outcome", "outcome.window", "split", "features.cr_last"):
        assert not gr.matches("cohort.subgroups", key)
    d, e = variant("dynamic", ["E02"])
    g = grade(d, e, [{"target": "cohort.subgroups", "problem": "ICU와 병동의 측정 빈도가 다르다"}])
    assert g.false_alarms == ["cohort.subgroups"] and not g.defects[0]["primary"]
    d, e = variant("dynamic")
    assert grade(d, e, [{"target": "cohort.subgroups", "problem": "측정 빈도"}]).false_alarms == ["cohort.subgroups"]


def test_h2_random_counts_subgroups_for_ascertainment():
    d, _ = variant("dynamic", ["E17"])
    uni = gr.universe(d)
    k = sum(gr.matches(u, "outcome.ascertainment") for u in uni)
    assert k == 2     # outcome, cohort.subgroups


def test_coarse_key_targets():
    d = BASES["dynamic"]
    assert gr.matches("cohort.inclusion.adult", gr.key_target("cohort", d))
    assert not gr.matches("features.age", gr.key_target("cohort", d))
    assert gr.matches("split", gr.key_target("model", d)) and gr.matches("model", gr.key_target("model", d))
    assert gr.key_target("cohort.inclusion:los_7d", d) == "cohort.inclusion.los_7d"


def test_selection_time_via_context():
    d, e = variant("dynamic", ["E14"])                      # Q5, cohort.inclusion.los_7d
    g = grade(d, e, [{"target": "cohort.inclusion.los_7d", "problem": "재원 기간은 퇴원 때 확정되는 미래 정보다"}])
    assert g.defects[0]["secondary"]


def test_two_defects_one_variant():
    d, e = variant("dynamic", ["E02", "E08"])
    g = grade(d, e, [{"target": "preprocessing.median_impute", "problem": "전체 데이터로 적합"}])
    assert [x["secondary"] for x in g.defects] == [False, True]


# ---------------------------------------------------------------- 점검기 출력 채점

def test_grade_checker_output():
    d, e = variant("dynamic", ["E02", "E13"])
    out = {"findings": [
        {"question": "Q1", "verdict": "차단", "target": "features.cr_last", "reason": "확인 가능 시각이 기준 시점(tp) 뒤일 수 있다"},
        {"question": "Q4", "verdict": "경고", "target": "features.renal_ordered", "reason": "대리 변수 목록과 일치"},
        {"question": "Q2", "verdict": "통과", "target": "설계 전체", "reason": "-"},
        {"question": "Q1~Q3", "verdict": "경고", "target": "features.age", "reason": "출처 불명: Q1~Q3 확인 불가."},
        {"question": "Q6", "verdict": "기록", "target": "attempts", "reason": "설계 1개"}]}
    g = gr.grade_checker({"report_id": "c1", "variant": "v.json", "exit_code": 1, "checker": out}, d, e)
    assert [x["secondary"] for x in g.defects] == [True, True]
    assert g.false_alarms == [] and g.n_unable == 1
    g2 = gr.grade_checker({"report_id": "c2", "variant": "v.json", "exit_code": 2, "checker": None}, d, e)
    assert g2.n_unable == 1 and not any(x["primary"] for x in g2.defects)


def test_checker_on_clean_base_has_no_false_alarm():
    """동결된 점검기의 실제 출력(설계서 단계)을 깨끗한 기본 설계에 채점하면 오경보 0."""
    from leakcheck.checks import run_checks
    for t in ("dynamic", "fixed"):
        d, e = variant(t)
        out = run_checks(d).to_dict()
        g = gr.grade_checker({"report_id": t, "variant": "v.json", "exit_code": 0, "checker": out}, d, e)
        assert g.false_alarms == [] and g.n_unresolved == 0


# ---------------------------------------------------------------- 맹검

def test_blind_guard():
    d, e = variant("dynamic", ["E02"])
    for k in ("condition", "arm", "조건"):
        with pytest.raises(ValueError):
            gr.grade_report({"report_id": "r", "variant": "v.json", "text": "", k: "가"}, d, e)
    with pytest.raises(ValueError):
        gr.grade_checker({"report_id": "r", "variant": "v.json", "exit_code": 0, "checker": None, "condition": "가"}, d, e)


def test_grade_all_has_no_condition_input():
    import inspect
    for fn in (gr.grade_report, gr.grade_checker, gr.grade_all):
        assert not {"condition", "conditions"} & set(inspect.signature(fn).parameters)


# ---------------------------------------------------------------- 요약 · H1 · H2

def _fake_run():
    """가짜 변형 4개 (결함 공개 1 · 결함 보류 1 · 깨끗한 것 2) × 조건 2 × 반복 3."""
    specs = {"vA.json": variant("dynamic", ["E02"]), "vB.json": variant("fixed", ["E06"]),
             "vC.json": variant("dynamic"), "vD.json": variant("fixed", legit="prior_admission_dx")}
    d, e = specs["vB.json"]
    specs["vB.json"] = (d, as_holdout(e))
    designs = {k: v[0] for k, v in specs.items()}
    key = {k: v[1] for k, v in specs.items()}
    good = {"vA.json": [{"target": "features.cr_last", "problem": "채취 시각이라 tₚ 이후 값"}],
            "vB.json": [{"target": "split.key", "problem": "같은 환자의 입원이 학습과 평가에 나뉜다"}],
            "vC.json": [], "vD.json": [{"target": "features.prior_dx_renal_cardiac", "problem": "이상하다"}]}
    weak = {"vA.json": [{"target": "features.cr_last", "problem": "좀 의심스럽다"}],
            "vB.json": [{"target": "features.age", "problem": "확인 필요"}],
            "vC.json": [{"target": "features.age", "problem": "확인 필요"}], "vD.json": []}
    reports, conds, i = [], {}, 0
    for cond, items in (("가", good), ("나", weak)):
        for rep in range(3):
            for v in specs:
                i += 1
                rid = f"r{i:03d}"
                reports.append({"report_id": rid, "variant": v, "text": report(items[v])})
                conds[rid] = {"condition": cond, "rep": rep}
    return reports, conds, key, designs


def test_summary_end_to_end():
    reports, conds, key, designs = _fake_run()
    grades = gr.grade_all(reports, key, designs=dict(designs))
    s = gr.summarize(grades, conds, key, dict(designs))
    ga, gb = s["conditions"]["가"], s["conditions"]["나"]
    assert ga["detection"]["all"]["secondary"] == 1.0 and gb["detection"]["all"]["secondary"] == 0.0
    assert gb["detection"]["all"]["primary"] == 0.5          # vA는 항목만 맞힘 (주 분석 탐지)
    assert ga["detection"]["holdout"]["secondary"] == 1.0 and ga["detection"]["public"]["n"] == 3
    # 오경보: 깨끗한·결함 변형 모두, 두 조건 모두
    assert ga["false_alarms"]["clean"] == {"reports": 6, "total": 3, "mean_per_report": 0.5}
    assert ga["false_alarms"]["defect"]["total"] == 0
    assert gb["false_alarms"]["clean"]["total"] == 3 and gb["false_alarms"]["defect"]["total"] == 3
    # 종류 불명 개수는 조건별로
    assert ga["counts"]["n_unknown_kind"] == 3 and gb["counts"]["n_unknown_kind"] == 9
    assert ga["stability"]["secondary"] == 1.0
    # H1: 차이 1.0, CI 하한 > 0, 깨끗한 변형 오경보 평균 0.5 ≤ 1
    assert s["H1"]["secondary"]["diff"] == 1.0 and s["H1"]["secondary"]["met"]
    # H2: (가)와 (나) 모두 보고
    h = s["H2"]["all_holdout"]
    assert h["가"]["secondary"]["observed"] == 1.0 and h["가"]["secondary"]["met"]
    assert 0 < h["가"]["secondary"]["expected_random"] < 1
    assert h["나"]["secondary"]["observed"] == 0.0
    assert s["H2"]["unadjusted_holdout"]["가"]["n"] == 3


def test_h1_fails_on_false_alarms_and_disagreement_flag():
    reports, conds, key, designs = _fake_run()
    noisy = [{"target": f"features.{n}", "problem": "의심"} for n in ("age", "sex", "unit")]
    for r in reports:
        if conds[r["report_id"]]["condition"] == "가" and r["variant"] in ("vC.json", "vD.json"):
            r["text"] = report(noisy)
    s = gr.summarize(gr.grade_all(reports, key, designs=dict(designs)), conds, key, dict(designs))
    assert s["conditions"]["가"]["false_alarms"]["clean"]["mean_per_report"] == 3.0
    assert s["H1"]["secondary"]["ci95"][0] > 0 and not s["H1"]["secondary"]["met"]
    # 주·보조 결론이 같으면 불일치 목록은 비어 있다
    assert "H1" not in s["primary_secondary_disagree"]


def test_disagreement_reported():
    reports, conds, key, designs = _fake_run()
    for r in reports:   # (가)가 종류를 말하지 않음 → 주 분석(항목) 차이 > 0, 보조 분석(항목+종류) 차이 0
        if conds[r["report_id"]]["condition"] == "가" and r["variant"] in ("vA.json", "vB.json"):
            tgt = "features.cr_last" if r["variant"] == "vA.json" else "split.key"
            r["text"] = report([{"target": tgt, "problem": "의심"}])
        elif conds[r["report_id"]]["condition"] == "나" and r["variant"] == "vA.json":
            r["text"] = report([])
    s = gr.summarize(gr.grade_all(reports, key, designs=dict(designs)), conds, key, dict(designs))
    assert not s["H1"]["secondary"]["met"] and s["H1"]["primary"]["met"]
    assert "H1" in s["primary_secondary_disagree"]


def test_checker_only_metric():
    reports, conds, key, designs = _fake_run()
    chk = [{"report_id": r["report_id"], "variant": r["variant"], "exit_code": 1,
            "checker": {"findings": [{"question": "Q1", "verdict": "차단", "target": "features.cr_last", "reason": "-"}]}}
           for r in reports if conds[r["report_id"]]["condition"] == "가"]
    s = gr.summarize(gr.grade_all(reports, key, chk, designs=dict(designs)), conds, key, dict(designs))
    c = s["checker_only"]
    assert c["runs"] == 12 and c["detection"]["secondary"] == 0.5     # vA만 맞힘
    assert c["false_alarms"]["clean"]["total"] == 6                   # vC·vD에서 cr_last 지적


def test_random_hit_prob():
    assert gr.random_hit_prob(10, 1, 0) == (0.0, 0.0)
    item_only, item_kind = gr.random_hit_prob(10, 1, 1)     # (주 분석, 보조 분석)
    assert item_only == pytest.approx(0.1) and item_kind == pytest.approx(0.1 / 6)
    assert gr.random_hit_prob(10, 1, 10)[0] == pytest.approx(1.0)
    assert gr.random_hit_prob(10, 2, 3)[0] == pytest.approx(1 - (8 * 7 * 6) / (10 * 9 * 8))


def test_universe_on_real_variants():
    """실제 맹검 변형 20개에서 항목 목록을 만들 수 있다 (정답표는 쓰지 않는다)."""
    for p in sorted(gr.VARIANT_DIR.glob("design_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        u = gr.universe(d)
        assert len(u) == len(set(u)) and all(gr.resolve(i, d) == i for i in u)


# ---------------------------------------------------------------- 변형 해시 대조

def test_verify_variants(tmp_path):
    log = gr.injection_log_hashes()
    assert len(log) == 20
    fake_key = {f: {"sha256": h} for f, h in log.items()}
    assert gr.verify_variants(fake_key) == []
    for f in log:
        (tmp_path / f).write_text((gr.VARIANT_DIR / f).read_text(encoding="utf-8"), encoding="utf-8")
    first = sorted(log)[0]
    (tmp_path / first).write_text("{}\n", encoding="utf-8")
    bad = gr.verify_variants(fake_key, tmp_path)
    assert len(bad) == 2 and all(first in b for b in bad)


# ---------------------------------------------------------------- 지시문

def test_prompt_fixed_and_neutral():
    t = prompt.text()
    assert prompt.PROMPT_SHA256 in (ROOT / "docs" / "success_criteria.md").read_text(encoding="utf-8")
    assert "한국어" in t and "```findings" in t
    banned = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "가용 시점", "독립 단위", "적합 범위", "결과 출처",
              "선택 시점", "확인 균질성", "누수", "leak", "스킬", "skill", "점검기", "run_check"]
    assert not [w for w in banned if w.lower() in t.lower()]
    r = prompt.render("design_0000.json", "data/synth")
    assert "{{" not in r and "design_0000.json" in r and "data/synth" in r


def test_prompt_example_block_parses():
    items, err, bad = gr.parse_findings(prompt.text())
    assert err is None and bad == 0 and len(items) == 1


def test_locked_files_match_success_criteria():
    """5단계에 고정한 지시문·채점기·채점 규칙이 바뀌지 않았다."""
    import hashlib
    doc = (ROOT / "docs" / "success_criteria.md").read_text(encoding="utf-8")
    locked = dict(re.findall(r"^\| (experiment/\S+) \| `([0-9a-f]{64})` \|", doc, re.M))
    assert set(locked) == {"experiment/agent_prompt.md", "experiment/grader.py", "experiment/scoring_rules.md"}
    for f, h in locked.items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h, f
