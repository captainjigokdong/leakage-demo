#!/usr/bin/env python3
"""설계서 누수 점검 실행기.

사용:
  python run_check.py 설계서.json                       # 설계서만 점검
  python run_check.py 설계서.json --data data/synth     # 데이터로 행 단위 재확인까지
  python run_check.py 설계서.json --data data/synth --card   # 결정 카드(기술 통계만) 출력
  python run_check.py 설계서.json --json 결과.json       # 구조화된 결과 저장

종료 코드: 0 차단 없음 · 1 차단 있음 · 2 점검 불가(형식 오류, 꼬리표 오류, 패키지 없음)

leakcheck 패키지 위치: 환경 변수 LEAKCHECK_HOME, 없으면 현재 폴더와 이 스크립트에서 위로 찾는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _find_home() -> Path | None:
    env = os.environ.get("LEAKCHECK_HOME")
    if env:
        return Path(env)
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for p in (start, *start.parents):
            if (p / "leakcheck" / "__init__.py").exists():
                return p
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="설계서 누수 점검 (Q1~Q7)")
    ap.add_argument("design", type=Path)
    ap.add_argument("--data", type=Path, help="합성 EHR 폴더 (*.csv.gz)")
    ap.add_argument("--json", type=Path, help="결과 JSON 저장 경로")
    ap.add_argument("--card", action="store_true", help="결정 카드 출력 (--data 필요)")
    args = ap.parse_args(argv)

    home = _find_home()
    if home is None:
        print("점검 불가: leakcheck 패키지를 찾지 못했다. LEAKCHECK_HOME을 저장소 루트로 지정하라.", file=sys.stderr)
        return 2
    sys.path.insert(0, str(home))

    from leakcheck.checks import run_checks
    from leakcheck.design import DesignError, load
    from leakcheck.tagging import TaggingError

    try:
        design = load(args.design)
        tables = None
        if args.data:
            from synth.generate import load as load_tables
            tables = load_tables(args.data)
        report = run_checks(design, tables)
    except (DesignError, TaggingError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"점검 불가: {e}", file=sys.stderr)
        return 2

    print(report.to_text())
    if args.json:
        args.json.write_text(report.to_json() + "\n", encoding="utf-8")
    else:
        print("\n--- JSON ---")
        print(json.dumps(report.to_dict()["findings"], ensure_ascii=False, indent=2))

    if args.card:
        if tables is None:
            print("결정 카드에는 --data가 필요하다.", file=sys.stderr)
            return 2
        if report.blocked:
            print("\n차단이 있어 결정 카드를 만들지 않는다. 설계를 먼저 고친다.")
        else:
            from leakcheck.lock import decision_card
            print("\n" + decision_card(design, tables).to_text())
    return 1 if report.blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
