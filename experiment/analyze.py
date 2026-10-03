"""7단계 채점과 분석 (docs/step7_analysis_plan.md 2·3절).

잠긴 채점기(experiment/grader.py)의 함수를 그대로 불러 쓴다. 채점 규칙은 여기서 바꾸지 않는다.
세 분석 묶음을 같은 방식으로 채점·요약한다.
  main   주 분석: 채택된 보고서 (빈칸 행은 분모에서 빠짐)
  first  첫 시도 분석: 행마다 첫 시도 보고서 (폐기된 시도 포함)
  blank  빈칸 = 미탐지 분석: 채택된 보고서 + 빈칸 행을 "지적 0개" 보고서로

  python -m experiment.analyze        (암호: 환경 변수 또는 터미널 입력, 정답표는 메모리에서만 복호화)
"""
from __future__ import annotations

import json
from pathlib import Path

from experiment import grader

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiment" / "runs"
RESULTS = ROOT / "results"
EMPTY_REPORT = "```findings\n[]\n```"
SETS = ("main", "first", "blank")


def _read(d: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def blank_rows(conditions: dict[str, dict], reports: list[dict]) -> list[str]:
    have = {r["report_id"] for r in reports}
    return sorted(rid for rid in conditions if rid not in have)


def fill_blanks(conditions: dict[str, dict], reports: list[dict]) -> list[dict]:
    """빈칸 행을 지적 0개 보고서로 채운다 (모든 배치 미탐지, 오경보 0)."""
    return reports + [{"report_id": rid, "variant": conditions[rid]["variant"], "text": EMPTY_REPORT}
                      for rid in blank_rows(conditions, reports)]


def report_sets(conditions: dict[str, dict], runs: Path = RUNS) -> dict[str, list[dict]]:
    reports = _read(runs / "reports")
    return {"main": reports, "first": _read(runs / "first_attempt_reports"),
            "blank": fill_blanks(conditions, reports)}


def invoked_flags(conditions: dict[str, dict], runs: Path = RUNS) -> dict[str, dict[str, bool]]:
    """행별 checker_invoked: 채택(마지막) 시도와 첫 시도. 빈칸 행은 채택 없음."""
    out = {}
    for rid in conditions:
        m = json.loads((runs / "meta" / f"{rid}.json").read_text(encoding="utf-8"))
        att = m["attempts"]
        out[rid] = {"first": bool(att[0]["audit"]["checker_invoked"]),
                    "adopted": bool(att[-1]["audit"]["checker_invoked"]) if m["status"] == "done" else None,
                    "no_checker_after_denial": bool(att[-1].get("no_checker_after_denial")) if m["status"] == "done" else None}
    return out


def invoked_split(grades: list[dict], conditions: dict[str, dict], flags: dict[str, dict], which: str) -> dict:
    """(가)를 점검기를 부른 실행 / 부르지 않은 실행으로 나눈 탐지율 (참고 분석)."""
    ga = [g for g in grades if g["source"] == "report" and conditions[g["report_id"]]["condition"] == "가"]
    out = {}
    for label, want in (("invoked", True), ("not_invoked", False)):
        gs = [g for g in ga if flags[g["report_id"]][which] is want]
        out[label] = {"reports": len(gs), "detection": grader.detection(gs),
                      "holdout": grader.detection(gs, lambda d: d["holdout"])}
    out["not_invoked_with_denial"] = sum(1 for g in ga if flags[g["report_id"]][which] is False
                                         and flags[g["report_id"]]["no_checker_after_denial"])
    return out


def overview_row(summary: dict) -> dict:
    """보고서 맨 앞 표의 한 줄: H1·H2의 주·보조 분석 기준 충족 여부와 수치."""
    h1, h2 = summary["H1"], summary["H2"]
    row = {}
    for meas in ("primary", "secondary"):
        row[meas] = {
            "H1_diff": h1[meas]["diff"], "H1_ci95": h1[meas]["ci95"], "H1_fa_clean_ga": h1[meas]["fa_clean_mean_ga"],
            "H1_met": h1[meas]["met"],
            "detection": {c: summary["conditions"][c]["detection"]["all"][meas] for c in ("가", "나")},
        }
        for scope in ("all_holdout", "unadjusted_holdout"):
            row[meas][f"H2_{scope}"] = {c: {k: h2[scope][c].get(meas, {}).get(k)
                                            for k in ("observed", "expected_random", "met", "p_one_sided")}
                                        for c in ("가", "나")}
        row[meas]["checker_only_holdout"] = summary["checker_only"]["holdout"][meas]
    row["unknown_kind"] = {c: summary["conditions"][c]["counts"]["n_unknown_kind"] for c in ("가", "나")}
    row["reports"] = {c: summary["conditions"][c]["reports"] for c in ("가", "나")}
    return row


def run(password: str, out: Path = RESULTS, runs: Path = RUNS) -> dict:
    key = grader.load_answer_key(password)
    bad = grader.verify_variants(key["variants"])
    if bad:
        raise SystemExit("변형 파일 해시 불일치:\n" + "\n".join(bad))
    conditions = json.loads((runs / "conditions.json").read_text(encoding="utf-8"))
    checker = _read(runs / "checker")
    flags = invoked_flags(conditions, runs)
    out.mkdir(parents=True, exist_ok=True)
    overview = {"blank_rows": {rid: conditions[rid]["condition"] for rid in blank_rows(conditions, _read(runs / "reports"))},
                "sets": {}}
    for name, reports in report_sets(conditions, runs).items():
        grades = grader.grade_all(reports, key["variants"], checker)
        summary = grader.summarize(grades, conditions, key["variants"])
        if name in ("main", "first"):
            summary["invoked_split"] = invoked_split(grades, conditions, flags, "adopted" if name == "main" else "first")
        (out / f"grades_{name}.json").write_text(json.dumps(grades, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out / f"summary_{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        overview["sets"][name] = overview_row(summary)
    overview["variant_hashes_verified"] = True
    (out / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return overview


def main() -> int:
    from tools.seal import get_password
    print(json.dumps(run(get_password()), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
