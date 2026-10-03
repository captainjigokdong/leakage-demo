"""7단계 분석 스크립트 시험. 진짜 정답표는 쓰지 않는다 (test_grader의 가짜 실행 재사용)."""
from __future__ import annotations

import json

from experiment import analyze as an
from experiment import grader as gr
from tests.test_grader import _fake_run as _grader_fake_run


def _fake_run():
    """test_grader의 가짜 실행 + 실제 조건표처럼 variant 칸."""
    reports, conds, key, designs = _grader_fake_run()
    for r in reports:
        conds[r["report_id"]]["variant"] = r["variant"]
    return reports, conds, key, designs


def test_blank_row_counts_as_miss_without_false_alarms():
    reports, conds, key, designs = _fake_run()
    dropped = next(r for r in reports if conds[r["report_id"]]["condition"] == "가" and key[r["variant"]]["defects"])
    kept = [r for r in reports if r is not dropped]
    assert an.blank_rows(conds, kept) == [dropped["report_id"]]
    filled = an.fill_blanks(conds, kept)
    assert len(filled) == len(reports)
    g = next(x for x in gr.grade_all(filled, key, designs=dict(designs)) if x["report_id"] == dropped["report_id"])
    assert g["format_error"] is None and g["false_alarms"] == []
    assert g["defects"] and not any(d["primary"] or d["secondary"] for d in g["defects"])


def test_blank_analysis_detection_not_higher_than_main():
    reports, conds, key, designs = _fake_run()
    kept = [r for r in reports if not (conds[r["report_id"]]["condition"] == "가" and r["variant"] == "vA.json")]
    s_main = gr.summarize(gr.grade_all(kept, key, designs=dict(designs)), conds, key, dict(designs))
    s_blank = gr.summarize(gr.grade_all(an.fill_blanks(conds, kept), key, designs=dict(designs)), conds, key, dict(designs))
    a, b = s_main["conditions"]["가"]["detection"]["all"], s_blank["conditions"]["가"]["detection"]["all"]
    assert b["n"] > a["n"] and b["primary"] <= a["primary"]


def test_overview_row_and_invoked_split():
    reports, conds, key, designs = _fake_run()
    grades = gr.grade_all(reports, key, designs=dict(designs))
    s = gr.summarize(grades, conds, key, dict(designs))
    row = an.overview_row(s)
    for meas in ("primary", "secondary"):
        assert set(row[meas]["H2_all_holdout"]) == {"가", "나"}
        assert isinstance(row[meas]["H1_met"], bool)
    assert row["unknown_kind"] == {"가": 3, "나": 9}
    ga = [rid for rid, c in conds.items() if c["condition"] == "가"]
    flags = {rid: {"adopted": rid != ga[0], "first": True, "no_checker_after_denial": rid == ga[0]} for rid in conds}
    sp = an.invoked_split(grades, conds, flags, "adopted")
    assert sp["not_invoked"]["reports"] == 1 and sp["invoked"]["reports"] == len(ga) - 1
    assert sp["not_invoked_with_denial"] == 1


def test_real_run_files_without_key():
    """실제 실행 기록의 구조만 확인 (정답표 불필요): 첫 시도 120개, 빈칸 1행, 모든 행에 checker_invoked."""
    conds = json.loads((an.RUNS / "conditions.json").read_text(encoding="utf-8"))
    sets = an.report_sets(conds)
    assert len(sets["first"]) == len(conds) == len(sets["blank"])
    assert len(an.blank_rows(conds, sets["main"])) == 1
    flags = an.invoked_flags(conds)
    assert set(flags) == set(conds)
    assert all(flags[r]["adopted"] is None for r in an.blank_rows(conds, sets["main"]))
