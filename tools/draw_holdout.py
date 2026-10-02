"""질문별 층화 보류 추첨 (1단계).

오류 사례 18개 전체 목록(평문, 저장소 밖 임시 경로)을 읽어 질문(Q)마다 1개씩
무작위로 보류한다.
- 공개 12개 → designs/error_catalog_public.csv
- 보류 6개 → 메모리에서 바로 암호화 → sealed/holdout.enc (평문 파일을 만들지 않음)
- 시드·날짜·배분·전체 목록 해시 → docs/holdout_log.md

이 파일에는 사례 내용이 없다. 암호는 환경 변수 SEAL_PASSWORD로 받는다.

사용 예:
    SEAL_PASSWORD=... python -m tools.draw_holdout /임시경로/catalog_full.plain.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import random
import secrets
from collections import Counter
from pathlib import Path

from tools.seal import decrypt_bytes, encrypt_bytes, get_password

COLUMNS = ["id", "name", "source", "question", "design_types", "how_to_inject", "expected_verdict"]
# docs/research_plan.md 7.1 권장 배분 (Q6는 기록만 하므로 사례 없음)
ALLOCATION = {"Q1": 4, "Q2": 3, "Q3": 3, "Q4": 3, "Q5": 3, "Q7": 2}
HOLDOUT_PER_QUESTION = 1
DESIGN_TYPES = {"고정", "동적", "둘다"}
VERDICTS = {"차단", "경고"}

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_CSV = ROOT / "designs" / "error_catalog_public.csv"
SEALED = ROOT / "sealed" / "holdout.enc"
LOG = ROOT / "docs" / "holdout_log.md"


def read_catalog(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"열이 다르다: {reader.fieldnames} (기대: {COLUMNS})")
        rows = [dict(r) for r in reader]
    validate(rows)
    return rows


def validate(rows: list[dict]) -> None:
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("id 중복")
    for r in rows:
        if any(not r[c].strip() for c in COLUMNS):
            raise ValueError(f"빈 칸이 있다: {r['id']}")
        if r["design_types"] not in DESIGN_TYPES:
            raise ValueError(f"design_types 오류: {r['id']}")
        if r["expected_verdict"] not in VERDICTS:
            raise ValueError(f"expected_verdict 오류: {r['id']}")
    counts = Counter(r["question"] for r in rows)
    if dict(counts) != ALLOCATION:
        raise ValueError(f"질문별 배분이 계획과 다르다: {dict(counts)} (기대: {ALLOCATION})")


def catalog_hash(rows: list[dict]) -> str:
    """전체 목록의 정규화 해시. 공개 CSV + 보류 행으로 나중에 다시 계산할 수 있다."""
    canon = json.dumps(sorted(rows, key=lambda r: r["id"]), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def stratified_draw(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """질문마다 id 순으로 정렬한 뒤 HOLDOUT_PER_QUESTION개씩 보류한다."""
    rng = random.Random(seed)
    held_ids: set[str] = set()
    for q in sorted(ALLOCATION):
        stratum = sorted((r for r in rows if r["question"] == q), key=lambda r: r["id"])
        held_ids.update(r["id"] for r in rng.sample(stratum, HOLDOUT_PER_QUESTION))
    public = sorted((r for r in rows if r["id"] not in held_ids), key=lambda r: r["id"])
    holdout = sorted((r for r in rows if r["id"] in held_ids), key=lambda r: r["id"])
    return public, holdout


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_log(seed: int, date: str, digest: str, public: list[dict]) -> None:
    pub_counts = Counter(r["question"] for r in public)
    lines = [
        "# 보류 추첨 기록",
        "",
        "`tools/draw_holdout.py`로 질문별 층화 추첨을 했다 (질문마다 1개 보류).",
        "보류 사례는 `sealed/holdout.enc`에만 있다. 4단계 전에는 열지 않는다.",
        "",
        f"- 날짜: {date}",
        f"- 시드: {seed}",
        "- 방법: 질문별로 id 순 정렬 → `random.Random(seed).sample(층, 1)`",
        f"- 전체 18개 목록 SHA-256 (id 순 정렬 JSON, `catalog_hash`): `{digest}`",
        "  - 보류 해제 후 공개 CSV + 보류 행으로 다시 계산해 목록이 바뀌지 않았음을 확인한다.",
        "",
        "| 질문 | 전체 | 보류 | 공개 |",
        "|---|---|---|---|",
    ]
    for q in sorted(ALLOCATION):
        lines.append(f"| {q} | {ALLOCATION[q]} | {HOLDOUT_PER_QUESTION} | {pub_counts[q]} |")
    lines.append(f"| 합계 | {sum(ALLOCATION.values())} | {HOLDOUT_PER_QUESTION * len(ALLOCATION)} | {len(public)} |")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("catalog", type=Path, help="18개 전체 목록 (저장소 밖 임시 경로)")
    args = p.parse_args(argv)

    if ROOT in args.catalog.resolve().parents:
        raise SystemExit("전체 목록은 저장소 밖에 두어야 한다.")
    if SEALED.exists():
        raise SystemExit(f"{SEALED}가 이미 있다. 추첨은 한 번만 한다.")

    rows = read_catalog(args.catalog)
    password = get_password()
    seed = secrets.randbits(32)
    date = dt.date.today().isoformat()
    digest = catalog_hash(rows)
    public, holdout = stratified_draw(rows, seed)

    payload = {
        "seed": seed,
        "date": date,
        "catalog_sha256": digest,
        "all_ids": sorted(r["id"] for r in rows),
        "holdout": holdout,
    }
    blob = encrypt_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), password)
    # 봉인 직후 메모리에서 복호화해 확인
    assert json.loads(decrypt_bytes(blob, password)) == payload

    PUBLIC_CSV.write_text(to_csv(public), encoding="utf-8")
    SEALED.write_bytes(blob)
    write_log(seed, date, digest, public)
    print(f"공개 {len(public)}개 → {PUBLIC_CSV.relative_to(ROOT)}")
    print(f"보류 {len(holdout)}개 → {SEALED.relative_to(ROOT)} (암호화)")
    print(f"시드 {seed}, 날짜 {date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
