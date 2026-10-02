"""합성 EHR의 기술 통계와 함정별 개수.

모델 성능은 계산하지 않는다 (CLAUDE.md 절대 규칙 6). 여기서 계산하는 것은
결과율·표본 수 같은 기술 통계와, 데이터에 함정이 충분히 들어 있는지 세는 개수뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

KDIGO_ABS_RISE = 0.3  # mg/dL, 48시간 안
KDIGO_ABS_WINDOW_H = 48
KDIGO_REL_RISE = 1.5  # 기저치 대비 배수, 7일 안
KDIGO_REL_WINDOW_H = 7 * 24
READMIT_DAYS = 30
EPS = 1e-9

# phases.md 2단계 완료 기준
TARGET_RATE = (0.10, 0.20)
MIN_MULTI_ADMIT_FRAC = 0.20
MIN_POST_DISCHARGE_LAB_FRAC = 0.02
MIN_FAMILY_PAIRS = 30


def kdigo_first(hours: np.ndarray, values: np.ndarray) -> int | None:
    """크레아티닌 측정(채취 시각 순 정렬)에서 KDIGO 기준을 처음 만족한 측정의 위치.

    기준 (어느 하나):
    - 앞선 48시간 안의 어떤 측정보다 0.3 mg/dL 이상 높음
    - 앞선 7일 안의 최솟값(기저치)의 1.5배 이상
    같은 입원 안의 측정만 본다. 만족하는 측정이 없으면 None.
    """
    for j in range(1, len(values)):
        prior = hours[:j] >= hours[j] - KDIGO_REL_WINDOW_H
        if not prior.any():
            continue
        if values[j] >= KDIGO_REL_RISE * values[:j][prior].min() - EPS:
            return j
        recent = hours[:j] >= hours[j] - KDIGO_ABS_WINDOW_H
        if recent.any() and values[j] - values[:j][recent].min() >= KDIGO_ABS_RISE - EPS:
            return j
    return None


def kdigo_table(labs: pd.DataFrame) -> pd.DataFrame:
    """입원별 KDIGO AKI 판정. 열: admission_id, aki, aki_collect_time, aki_report_time."""
    cr = labs[labs["test"] == "creatinine"].sort_values(["admission_id", "collect_time"])
    rows = []
    for adm, g in cr.groupby("admission_id", sort=True):
        hours = (g["collect_time"] - g["collect_time"].iloc[0]).dt.total_seconds().to_numpy() / 3600
        j = kdigo_first(hours, g["value"].to_numpy(dtype=float))
        if j is None:
            rows.append((adm, False, pd.NaT, pd.NaT))
        else:
            rows.append((adm, True, g["collect_time"].iloc[j], g["report_time"].iloc[j]))
    return pd.DataFrame(rows, columns=["admission_id", "aki", "aki_collect_time", "aki_report_time"])


def readmission_table(admissions: pd.DataFrame) -> pd.DataFrame:
    """생존 퇴원한 입원별 30일 내 재입원 여부. 열: admission_id, patient_id, readmit30."""
    a = admissions.sort_values(["patient_id", "admit_time"]).copy()
    a["next_admit"] = a.groupby("patient_id")["admit_time"].shift(-1)
    a = a[a["discharge_status"] != "died"]
    gap = a["next_admit"] - a["discharge_time"]
    a["readmit30"] = gap.notna() & (gap <= pd.Timedelta(days=READMIT_DAYS))
    return a[["admission_id", "patient_id", "readmit30"]].reset_index(drop=True)


@dataclass
class Check:
    name: str
    value: str
    criterion: str
    passed: bool


def summarize(tables: dict[str, pd.DataFrame]) -> dict:
    pts, adm, labs, orders = tables["patients"], tables["admissions"], tables["labs"], tables["orders"]

    kd = kdigo_table(labs)
    adm_aki = adm.merge(kd, on="admission_id", how="left")
    adm_aki["aki"] = adm_aki["aki"].fillna(False).astype(bool)
    rd = readmission_table(adm)

    n_adm_per_pt = adm.groupby("patient_id").size().reindex(pts["patient_id"], fill_value=0)

    lab_dis = labs.merge(adm[["admission_id", "discharge_time"]], on="admission_id")
    post_dis = (lab_dis["collect_time"] <= lab_dis["discharge_time"]) & (
        lab_dis["report_time"] > lab_dis["discharge_time"]
    )

    fam_sizes = pts["family_id"].dropna().value_counts()
    family_pairs = int(sum(comb(int(s), 2) for s in fam_sizes))

    renal = orders[orders["order_type"].isin(["dialysis_order", "nephrology_consult"])]
    aki_adm = adm_aki[adm_aki["aki"]]
    aki_renal_pts = aki_adm[aki_adm["admission_id"].isin(renal["admission_id"])]["patient_id"].nunique()

    per_day = labs[labs["test"] == "creatinine"].groupby("admission_id").size().rename("n_cr")
    los_days = (adm["discharge_time"] - adm["admit_time"]).dt.total_seconds() / 86400
    freq = adm.assign(los_days=los_days).merge(per_day, on="admission_id", how="left").fillna({"n_cr": 0})
    freq["cr_per_day"] = freq["n_cr"] / freq["los_days"]

    return {
        "n_patients": len(pts),
        "n_admissions": len(adm),
        "n_labs": len(labs),
        "n_diagnoses": len(tables["diagnoses"]),
        "n_procedures": len(tables["procedures"]),
        "n_orders": len(orders),
        "admissions_per_patient_mean": float(n_adm_per_pt.mean()),
        "admissions_per_patient_dist": n_adm_per_pt.value_counts().sort_index().to_dict(),
        "aki_rate": float(adm_aki["aki"].mean()),
        "aki_rate_icu": float(adm_aki.loc[adm_aki["unit"] == "ICU", "aki"].mean()),
        "aki_rate_ward": float(adm_aki.loc[adm_aki["unit"] == "ward", "aki"].mean()),
        "readmit30_rate": float(rd["readmit30"].mean()),
        "death_rate": float((adm["discharge_status"] == "died").mean()),
        "multi_admit_frac": float((n_adm_per_pt >= 2).mean()),
        "post_discharge_lab_frac": float(post_dis.mean()),
        "post_discharge_lab_n": int(post_dis.sum()),
        "family_pairs": family_pairs,
        "n_families": int(len(fam_sizes)),
        "icu_patients": int(adm.loc[adm["unit"] == "ICU", "patient_id"].nunique()),
        "ward_patients": int(adm.loc[adm["unit"] == "ward", "patient_id"].nunique()),
        "aki_patients_with_renal_order": int(aki_renal_pts),
        "cr_per_day_icu": float(freq.loc[freq["unit"] == "ICU", "cr_per_day"].mean()),
        "cr_per_day_ward": float(freq.loc[freq["unit"] == "ward", "cr_per_day"].mean()),
    }


def check_criteria(s: dict) -> list[Check]:
    lo, hi = TARGET_RATE
    pct = lambda x: f"{x:.1%}"  # noqa: E731
    return [
        Check("AKI 비율 (입원 기준)", pct(s["aki_rate"]), "10~20%", lo <= s["aki_rate"] <= hi),
        Check("30일 재입원 비율 (생존 퇴원 기준)", pct(s["readmit30_rate"]), "10~20%",
              lo <= s["readmit30_rate"] <= hi),
        Check("입원 2회 이상 환자", pct(s["multi_admit_frac"]), ">= 20%",
              s["multi_admit_frac"] >= MIN_MULTI_ADMIT_FRAC),
        Check("퇴원 직전 채취·퇴원 후 보고 검사",
              f"{pct(s['post_discharge_lab_frac'])} ({s['post_discharge_lab_n']}건)", ">= 2%",
              s["post_discharge_lab_frac"] >= MIN_POST_DISCHARGE_LAB_FRAC),
        Check("가족 관계 쌍", f"{s['family_pairs']}쌍 ({s['n_families']}가족)", ">= 30쌍",
              s["family_pairs"] >= MIN_FAMILY_PAIRS),
        Check("ICU 환자 / 병동 환자", f"{s['icu_patients']}명 / {s['ward_patients']}명", "둘 다 > 0",
              s["icu_patients"] > 0 and s["ward_patients"] > 0),
        Check("투석 오더·신장내과 협진이 있는 AKI 환자", f"{s['aki_patients_with_renal_order']}명", ">= 1",
              s["aki_patients_with_renal_order"] >= 1),
    ]


def format_report(s: dict, checks: list[Check]) -> str:
    dist = ", ".join(f"{k}회 {v}명" for k, v in s["admissions_per_patient_dist"].items())
    lines = [
        "== 기본 통계 ==",
        f"환자 {s['n_patients']}명, 입원 {s['n_admissions']}건, 검사 {s['n_labs']}행, "
        f"진단 {s['n_diagnoses']}행, 처치 {s['n_procedures']}행, 오더 {s['n_orders']}행",
        f"환자당 입원 수: 평균 {s['admissions_per_patient_mean']:.2f} ({dist})",
        f"AKI 비율: {s['aki_rate']:.1%} (ICU {s['aki_rate_icu']:.1%}, 병동 {s['aki_rate_ward']:.1%})",
        f"30일 재입원 비율: {s['readmit30_rate']:.1%}",
        f"원내 사망: {s['death_rate']:.1%}",
        f"크레아티닌 측정 빈도(회/일): ICU {s['cr_per_day_icu']:.2f}, 병동 {s['cr_per_day_ward']:.2f}",
        "",
        "== 함정별 개수 (완료 기준) ==",
    ]
    for c in checks:
        lines.append(f"[{'✔' if c.passed else '✘'}] {c.name}: {c.value}  (기준 {c.criterion})")
    return "\n".join(lines)
