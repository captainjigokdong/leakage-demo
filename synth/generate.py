"""합성 EHR 생성기 (2단계).

MIMIC-IV와 비슷한 구조의 가상 데이터를 만든다. 모든 환자·수치는 가상 값이고,
날짜는 MIMIC처럼 먼 미래(2150년대)로 옮겨 실제 날짜와 섞이지 않게 했다.

테이블
- patients    (patient_id, family_id, age, sex)
- admissions  (admission_id, patient_id, admit_time, discharge_time, discharge_status, unit)
- labs        (lab_id, admission_id, test, value, unit, collect_time, report_time)
- diagnoses   (admission_id, seq, icd_code)            ← 시각 없음
- procedures  (admission_id, code, chart_date)         ← 날짜만
- orders      (order_id, admission_id, order_type, order_time)

일부러 넣은 성질 (특정 오류 사례가 아니라 실제 EHR의 일반적 성질)
- 검사 보고 시각은 채취 30분~6시간 뒤. 퇴원 직전 채취·퇴원 후 보고 검사 포함
- ICU는 크레아티닌을 하루 1~2회, 병동은 2~3일에 1회 측정
- 가족은 잠재 위험을 공유
- AKI 정답은 측정된 크레아티닌에 KDIGO 기준을 적용해 정함 (잠재 AKI와 다를 수 있음)
- 신장내과 협진·투석 오더·투석 처치는 AKI 무렵에 몰림

난수는 numpy.random.default_rng(seed) 하나만 정해진 순서로 쓴다. 같은 시드 → 같은 데이터.

실행: python -m synth.generate [--n-patients 5000] [--seed 20261002] [--out data/synth]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from synth.stats import check_criteria, format_report, kdigo_first, summarize

GENERATOR_VERSION = "1"
DEFAULT_SEED = 20261002
DEFAULT_N_PATIENTS = 5000
DEFAULT_OUT = Path("data/synth")
TABLES = ("patients", "admissions", "labs", "diagnoses", "procedures", "orders")

EPOCH = pd.Timestamp("2150-01-01")
SPAN_DAYS = 10 * 365
H = 60  # 분 단위 시각에서 1시간
D = 24 * H
MAX_ADMISSIONS = 4

# 잠재 위험 모형의 절편 (기본 시드에서 AKI·30일 재입원 비율이 10~20%가 되게 맞춘 값)
AKI_INTERCEPT = -2.6
READMIT_INTERCEPT = -2.2
LATE_READMIT_P = 0.18

FAMILY_FRAC = 0.20
PRE_DISCHARGE_DRAW_P = 0.35

OTHER_DX = [  # (코드, 기본 확률)
    ("I10", 0.40), ("E11.9", 0.22), ("D64.9", 0.15), ("J18.9", 0.10),
    ("N39.0", 0.08), ("E87.1", 0.06), ("K21.9", 0.10), ("F32.9", 0.07),
]
OTHER_ORDERS = [  # (오더, 병동 확률, ICU 확률)
    ("chest_xray", 0.40, 0.90), ("physical_therapy_consult", 0.30, 0.40),
    ("social_work_consult", 0.12, 0.20), ("dietitian_consult", 0.10, 0.30),
    ("echocardiogram", 0.08, 0.25), ("ct_head", 0.04, 0.10),
]


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SynthEHR:
    tables: dict[str, pd.DataFrame]
    # 생성기 내부의 잠재 정답 (admission_id, aki_latent, onset_time, peak_mult). 파일로 저장하지 않는다.
    latent: pd.DataFrame
    seed: int
    n_patients: int


@dataclass
class _Rows:
    patients: list = field(default_factory=list)
    admissions: list = field(default_factory=list)
    labs: list = field(default_factory=list)
    diagnoses: list = field(default_factory=list)
    procedures: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    latent: list = field(default_factory=list)


def _assign_families(rng: np.random.Generator, n: int) -> list[str | None]:
    fam: list[str | None] = [None] * n
    order = rng.permutation(n)
    target = int(round(FAMILY_FRAC * n))
    pos, k = 0, 0
    while pos < target and pos + 2 <= n:
        size = int(rng.choice([2, 3], p=[0.6, 0.4]))
        size = min(size, n - pos)
        k += 1
        for i in order[pos:pos + size]:
            fam[i] = f"F{k:04d}"
        pos += size
    return fam


def _cr_true(h: float, base: float, onset: float | None, rise: float, peak: float) -> float:
    """잠재 크레아티닌 궤적. onset 전 기저치, rise 시간 동안 직선 상승, 이후 반감기 24시간으로 회복."""
    if onset is None or h <= onset:
        return base
    if h <= onset + rise:
        return base * (1 + (peak - 1) * (h - onset) / rise)
    floor = 1.1
    return base * (floor + (peak - floor) * 0.5 ** ((h - onset - rise) / 24))


def _draw_hours(rng: np.random.Generator, icu: bool, los_h: float) -> list[float]:
    """채혈 시각(입원 후 시간). ICU는 하루 1~2회, 병동은 40~72시간 간격."""
    hours = [float(rng.uniform(0.2, 2.0))]
    if icu:
        day = 0
        while True:
            for offset, p in ((6.0, 1.0), (18.0, 0.6)):
                h = day * 24 + offset + float(rng.normal(0, 0.7))
                if h > hours[-1] + 2 and rng.random() < p:
                    hours.append(h)
            day += 1
            if day * 24 > los_h:
                break
    else:
        while True:
            h = hours[-1] + float(rng.uniform(40, 72))
            if h > los_h:
                break
            hours.append(h)
    hours = [h for h in hours if h < los_h - 0.5]
    if rng.random() < PRE_DISCHARGE_DRAW_P:
        h = los_h - float(rng.uniform(0.5, 3.0))
        if not hours or h > hours[-1] + 1:
            hours.append(h)
    return hours


def _report_delay_min(rng: np.random.Generator, test: str) -> int:
    if test == "hemoglobin":
        m = np.exp(rng.normal(np.log(60), 0.5))
        return int(np.clip(round(m), 30, 240))
    m = np.exp(rng.normal(np.log(100), 0.6))
    return int(np.clip(round(m), 30, 360))


class _Generator:
    def __init__(self, seed: int, n_patients: int):
        self.rng = np.random.default_rng(seed)
        self.n = n_patients
        self.rows = _Rows()
        self.n_adm = self.n_lab = self.n_ord = 0

    def run(self) -> None:
        rng, n = self.rng, self.n
        age = np.clip(np.round(rng.normal(62, 16, n)), 18, 95).astype(int)
        sex = rng.choice(np.array(["F", "M"]), n)
        fam = _assign_families(rng, n)
        fam_effect = {f: float(rng.normal(0, 0.7)) for f in sorted({f for f in fam if f})}
        own = rng.normal(0, 0.7, n)
        indiv = rng.normal(0, 0.4, n)
        for i in range(n):
            pid = f"P{i + 1:05d}"
            z = (fam_effect[fam[i]] if fam[i] else own[i]) + indiv[i]
            self.rows.patients.append((pid, fam[i], int(age[i]), str(sex[i])))
            self._patient(pid, int(age[i]), str(sex[i]), z)

    def _patient(self, pid: str, age: int, sex: str, z: float) -> None:
        rng = self.rng
        ckd = rng.random() < _logistic(-2.2 + 0.04 * (age - 60) + 0.8 * z)
        base_cr = (0.95 if sex == "M" else 0.75) * float(np.exp(rng.normal(0, 0.12)))
        base_cr *= 1 + 0.004 * max(age - 40, 0)
        if ckd:
            base_cr *= float(rng.uniform(1.5, 2.5))
        t = int(rng.integers(0, (SPAN_DAYS - 400) * D))
        for _ in range(MAX_ADMISSIONS):
            discharge, alive, aki, icu = self._admission(pid, age, sex, z, ckd, base_cr, t)
            if not alive:
                break
            p30 = _logistic(READMIT_INTERCEPT + 0.6 * z + 0.9 * aki + 0.015 * (age - 60) + 0.3 * icu)
            if rng.random() < p30:
                gap = int(rng.integers(1 * D, 30 * D))
            elif rng.random() < LATE_READMIT_P:
                gap = int(rng.integers(31 * D, 720 * D))
            else:
                break
            t = discharge + gap
            if t > SPAN_DAYS * D:
                break

    def _admission(self, pid, age, sex, z, ckd, base_cr, admit) -> tuple[int, bool, bool, bool]:
        rng, R = self.rng, self.rows
        self.n_adm += 1
        aid = f"A{self.n_adm:06d}"

        icu = bool(rng.random() < _logistic(-1.3 + 0.5 * z + 0.01 * (age - 60)))
        los_h = float(np.clip(np.exp(rng.normal(np.log(110 if icu else 80), 0.5)), 24, 30 * 24))
        aki = bool(rng.random() < _logistic(AKI_INTERCEPT + 0.7 * z + 0.9 * icu + 0.7 * ckd + 0.02 * (age - 60)))
        onset = rise = None
        peak = 1.0
        if aki:
            onset = float(rng.uniform(12, max(13.0, min(0.7 * los_h, 168))))
            rise = float(rng.uniform(12, 48))
            peak = float(rng.uniform(1.3, 3.5))
            los_h = max(los_h, onset + rise + float(rng.uniform(36, 144)))
        los_min = int(round(los_h * H))
        discharge = admit + los_min
        base_adm = base_cr * float(np.exp(rng.normal(0, 0.06)))

        p_die = 0.01 + 0.03 * icu + 0.06 * (aki and peak > 2.5)
        u = rng.random()
        status = "died" if u < p_die else ("transfer" if u < p_die + 0.12 else "home")
        R.admissions.append((aid, pid, admit, discharge, status, "ICU" if icu else "ward"))
        R.latent.append((aid, aki, admit + int(round(onset * H)) if aki else None, peak))

        # 검사
        cr_obs: list[tuple[float, float, int]] = []  # (채취 시간, 값, 보고 시각)
        hgb_min, k_max = 99.0, 0.0
        for h in _draw_hours(rng, icu, los_h):
            collect = admit + int(round(h * H))
            true_cr = _cr_true(h, base_adm, onset, rise or 1.0, peak)
            cr = max(0.2, true_cr * float(np.exp(rng.normal(0, 0.03))) + float(rng.normal(0, 0.03)))
            cr = round(cr, 2)
            bun = max(3, round(10 + 9 * cr + float(rng.normal(0, 4)) + 4 * icu))
            k = round(4.1 + 0.4 * (true_cr / base_adm - 1) + float(rng.normal(0, 0.35)), 1)
            values = [("creatinine", cr, "mg/dL"), ("bun", bun, "mg/dL"), ("potassium", k, "mmol/L")]
            if rng.random() < 0.75:
                hgb = round((13.5 if sex == "M" else 12.2) - 1.2 * icu - 0.08 * h / 24
                            + float(rng.normal(0, 0.9)), 1)
                values.append(("hemoglobin", hgb, "g/dL"))
                hgb_min = min(hgb_min, hgb)
            k_max = max(k_max, k)
            for test, val, unit in values:
                report = collect + _report_delay_min(rng, test)
                self.n_lab += 1
                R.labs.append((f"L{self.n_lab:07d}", aid, test, val, unit, collect, report))
                if test == "creatinine":
                    cr_obs.append((h, cr, report))

        j = kdigo_first(np.array([c[0] for c in cr_obs]), np.array([c[1] for c in cr_obs]))
        observed = j is not None

        # 진단 (시각 없음)
        codes = []
        p_n17 = 0.9 if observed else (0.4 if aki else 0.01)
        if rng.random() < p_n17:
            codes.append("N17.9")
        if ckd and rng.random() < 0.9:
            codes.append("N18.3")
        if rng.random() < 0.05 + 0.15 * icu + 0.10 * aki:
            codes.append("A41.9")
        if rng.random() < 0.10 + 0.10 * icu:
            codes.append("I50.9")
        if k_max >= 5.5 and rng.random() < 0.7:
            codes.append("E87.5")
        for code, p in OTHER_DX:
            if rng.random() < p:
                codes.append(code)
        if not codes:
            codes.append("R53.83")
        for s, ci in enumerate(rng.permutation(len(codes)), start=1):
            R.diagnoses.append((aid, s, codes[ci]))

        # 처치 (날짜만)·오더 (시각)
        def day_of(minute: int) -> str:
            return (EPOCH + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%d")

        def order(kind: str, minute: int) -> None:
            if minute >= discharge:  # 퇴원 뒤로 넘어간 오더는 생기지 않은 것으로 본다
                return
            self.n_ord += 1
            R.orders.append((f"O{self.n_ord:07d}", aid, kind, int(max(minute, admit + 10))))

        if icu:
            if rng.random() < 0.5:
                R.procedures.append((aid, "02HV33Z", day_of(admit + int(rng.integers(0, D)))))
            if rng.random() < 0.25:
                R.procedures.append((aid, "0BH17EZ", day_of(admit + int(rng.integers(0, D)))))
        if hgb_min < 8 and rng.random() < 0.6 or rng.random() < 0.03:
            R.procedures.append((aid, "30233N1", day_of(admit + int(rng.integers(0, los_min)))))

        onset_min = admit + int(round(onset * H)) if aki else None
        if aki and (peak >= 2.5 or (ckd and peak >= 2.0)) and rng.random() < 0.45:
            start = onset_min + int(round((rise + float(rng.uniform(0, 24))) * H))
            if start < discharge - 6 * H:
                order("dialysis_order", start - int(rng.integers(1 * H, 6 * H)))
                for d in range(int(rng.integers(1, 4))):
                    if start + d * D < discharge:
                        R.procedures.append((aid, "5A1D70Z", day_of(start + d * D)))

        if observed:
            detect = cr_obs[j][2]
            if rng.random() < 0.6:
                if rng.random() < 0.25:  # 수치 보고 전에 임상적으로 의심해 협진
                    since = onset_min if aki else cr_obs[j - 1][2]
                    order("nephrology_consult", since + int(rng.integers(0, max(1, detect - since))))
                else:
                    order("nephrology_consult", detect + int(rng.integers(2 * H, 24 * H)))
        elif aki:
            if rng.random() < 0.2:
                order("nephrology_consult", onset_min + int(rng.integers(12 * H, 48 * H)))
        elif rng.random() < (0.12 if ckd else 0.03):
            order("nephrology_consult", admit + int(rng.integers(0, los_min)))

        if rng.random() < (0.4 if aki else 0.04):
            order("renal_ultrasound", (onset_min if aki else admit) + int(rng.integers(0, 36 * H)))
        for kind, p_ward, p_icu in OTHER_ORDERS:
            if rng.random() < (p_icu if icu else p_ward):
                order(kind, admit + int(rng.integers(0, los_min)))

        return discharge, status != "died", aki, icu

    def frames(self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        R = self.rows

        def ts(s: pd.Series) -> pd.Series:
            return EPOCH + pd.to_timedelta(s.astype("float64"), unit="min").dt.round("min")

        patients = pd.DataFrame(R.patients, columns=["patient_id", "family_id", "age", "sex"])
        adm = pd.DataFrame(R.admissions, columns=["admission_id", "patient_id", "admit_time",
                                                  "discharge_time", "discharge_status", "unit"])
        labs = pd.DataFrame(R.labs, columns=["lab_id", "admission_id", "test", "value", "unit",
                                             "collect_time", "report_time"])
        dx = pd.DataFrame(R.diagnoses, columns=["admission_id", "seq", "icd_code"])
        proc = pd.DataFrame(R.procedures, columns=["admission_id", "code", "chart_date"])
        proc = proc.sort_values(["admission_id", "chart_date", "code"], kind="stable").reset_index(drop=True)
        orders = pd.DataFrame(R.orders, columns=["order_id", "admission_id", "order_type", "order_time"])
        latent = pd.DataFrame(R.latent, columns=["admission_id", "aki_latent", "onset_time", "peak_mult"])

        for df, cols in ((adm, ["admit_time", "discharge_time"]), (labs, ["collect_time", "report_time"]),
                         (orders, ["order_time"]), (latent, ["onset_time"])):
            for c in cols:
                df[c] = ts(df[c])
        tables = {"patients": patients, "admissions": adm, "labs": labs,
                  "diagnoses": dx, "procedures": proc, "orders": orders}
        return tables, latent


def generate(seed: int = DEFAULT_SEED, n_patients: int = DEFAULT_N_PATIENTS) -> SynthEHR:
    if n_patients < 1:
        raise ValueError("n_patients는 1 이상이어야 한다")
    g = _Generator(seed, n_patients)
    g.run()
    tables, latent = g.frames()
    return SynthEHR(tables=tables, latent=latent, seed=seed, n_patients=n_patients)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S").encode("utf-8")


def save(ehr: SynthEHR, out_dir: Path) -> dict:
    """테이블을 <name>.csv.gz로 저장하고 MANIFEST.json(행 수, 압축 전 CSV의 sha256)을 쓴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"generator_version": GENERATOR_VERSION, "seed": ehr.seed,
                "n_patients": ehr.n_patients, "tables": {}}
    for name in TABLES:
        raw = to_csv_bytes(ehr.tables[name])
        with open(out_dir / f"{name}.csv.gz", "wb") as fh, gzip.GzipFile(
            filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9
        ) as gz:
            gz.write(raw)
        manifest["tables"][name] = {"file": f"{name}.csv.gz", "rows": len(ehr.tables[name]),
                                    "sha256_csv": hashlib.sha256(raw).hexdigest()}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


TIME_COLUMNS = {"admissions": ["admit_time", "discharge_time"], "labs": ["collect_time", "report_time"],
                "orders": ["order_time"]}


def load(out_dir: Path = DEFAULT_OUT) -> dict[str, pd.DataFrame]:
    """저장된 테이블을 읽는다. 시각 열은 datetime으로, chart_date는 문자열 그대로 둔다."""
    tables = {}
    for name in TABLES:
        df = pd.read_csv(out_dir / f"{name}.csv.gz", dtype={"family_id": "string", "chart_date": "string"})
        for c in TIME_COLUMNS.get(name, []):
            df[c] = pd.to_datetime(df[c])
        tables[name] = df
    return tables


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="합성 EHR 생성")
    ap.add_argument("--n-patients", type=int, default=DEFAULT_N_PATIENTS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-save", action="store_true", help="저장하지 않고 통계만 출력")
    args = ap.parse_args(argv)

    ehr = generate(args.seed, args.n_patients)
    if not args.no_save:
        save(ehr, args.out)
        print(f"저장: {args.out}/ (seed={args.seed}, n_patients={args.n_patients})\n")
    s = summarize(ehr.tables)
    checks = check_criteria(s)
    print(format_report(s, checks))
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
