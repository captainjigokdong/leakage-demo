"""외래 방문 테이블 (4단계 추가, 별도 파일).

기존 6개 테이블(synth.generate)은 그대로 두고, 저장된 입원 테이블에서 외래 방문 기록을 따로 만든다.
- 파일: data/synth/outpatient_visits.csv.gz (+ data/synth/OUTPATIENT_MANIFEST.json)
- 열: visit_id, patient_id, visit_time
- synth.generate.load()는 이 파일을 읽지 않는다. 기존 6개 테이블만 불러온다.
  (동결된 규칙표에 이 테이블의 규칙이 없어서, 섞어 넣으면 점검기가 전체를 멈춘다.)

일부러 넣은 성질 (실제 외래 기록의 일반적 성질)
- 생존 퇴원 뒤 추적 외래가 잡힌다. 자택 퇴원이 전원(transfer)보다 추적 외래가 잦다
- 추적 외래 전에 다시 입원하면 그 외래는 열리지 않는다
- 입원과 무관한 정기 외래도 있다

난수는 numpy.random.default_rng(seed) 하나만 정해진 순서로 쓴다. 같은 시드·같은 입원 테이블 → 같은 데이터.

실행: python -m synth.outpatient [--seed 20261003] [--data data/synth]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synth.generate import DEFAULT_OUT, EPOCH, SPAN_DAYS, load, to_csv_bytes

OUTPATIENT_VERSION = "1"
DEFAULT_SEED = 20261003
FILE = "outpatient_visits.csv.gz"
MANIFEST = "OUTPATIENT_MANIFEST.json"
COLUMNS = ["visit_id", "patient_id", "visit_time"]

FOLLOWUP_P = {"home": 0.70, "transfer": 0.35}  # 퇴원처별 추적 외래 예약 확률
FOLLOWUP_DAYS = (5, 28)                          # 퇴원 뒤 추적 외래까지 일수 범위
ROUTINE_RATE = 1.5                               # 환자당 정기 외래 수 평균 (포아송)


def generate_visits(patients: pd.DataFrame, admissions: pd.DataFrame,
                    seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, pd.Timestamp]] = []
    adm = admissions.sort_values(["patient_id", "admit_time", "admission_id"], kind="stable")
    for _, g in adm.groupby("patient_id", sort=True):
        admits = list(g["admit_time"])
        for i, a in enumerate(g.itertuples(index=False)):
            p = FOLLOWUP_P.get(a.discharge_status, 0.0)
            if not (p and rng.random() < p):
                continue
            when = a.discharge_time + pd.Timedelta(days=float(rng.uniform(*FOLLOWUP_DAYS)))
            nxt = admits[i + 1] if i + 1 < len(admits) else None
            if nxt is not None and nxt <= when:  # 다시 입원해 외래가 열리지 않음
                continue
            rows.append((a.patient_id, when))
    for pid in sorted(patients["patient_id"]):
        for _ in range(int(rng.poisson(ROUTINE_RATE))):
            rows.append((pid, EPOCH + pd.Timedelta(days=float(rng.uniform(0, SPAN_DAYS)))))
    df = pd.DataFrame(rows, columns=["patient_id", "visit_time"])
    df["visit_time"] = df["visit_time"].dt.round("min")
    df = df.sort_values(["patient_id", "visit_time"], kind="stable").reset_index(drop=True)
    df.insert(0, "visit_id", [f"V{i + 1:07d}" for i in range(len(df))])
    return df[COLUMNS]


def save(df: pd.DataFrame, out_dir: Path, seed: int) -> dict:
    raw = to_csv_bytes(df)
    with open(out_dir / FILE, "wb") as fh, gzip.GzipFile(
        filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9
    ) as gz:
        gz.write(raw)
    manifest = {"outpatient_version": OUTPATIENT_VERSION, "seed": seed,
                "built_from": "admissions.csv.gz, patients.csv.gz (MANIFEST.json)",
                "file": FILE, "rows": len(df), "sha256_csv": hashlib.sha256(raw).hexdigest()}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_visits(out_dir: Path = DEFAULT_OUT) -> pd.DataFrame:
    df = pd.read_csv(out_dir / FILE)
    df["visit_time"] = pd.to_datetime(df["visit_time"])
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="외래 방문 테이블 생성 (별도 파일)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--data", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    t = load(args.data)
    df = generate_visits(t["patients"], t["admissions"], args.seed)
    m = save(df, args.data, args.seed)
    print(f"저장: {args.data / FILE} ({m['rows']}행, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
