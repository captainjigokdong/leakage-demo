"""시험용 설계 변형.

공개 오류 사례 12개는 designs/error_catalog_public.csv의 how_to_inject를 깨끗한 설계에 적용한
것이다. 이 파일은 시험에만 쓰고, 점검 코드(leakcheck/)는 사례 ID를 전혀 모른다.
"수상해 보이지만 정당한" 변형은 원리상 통과해야 하는 특징들이다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
BASE = {"동적": "clean_dynamic_aki", "고정": "clean_fixed_readmission"}


def clean(kind: str) -> dict:
    return json.loads((FIXTURES / f"{BASE[kind]}.json").read_text(encoding="utf-8"))


def _feature(d, name):
    return next(f for f in d["features"] if f["name"] == name)


def _e01(d):
    d["features"].append({"name": "dx_this_admission", "source": "diagnoses",
                          "scope": "index_admission", "agg": "count"})


def _e02(d):
    _feature(d, "cr_last")["time_column"] = "collect_time"


def _e03(d):
    d["features"].append({"name": "cr_max_admission", "source": "labs", "filter": {"test": ["creatinine"]},
                          "time_column": "report_time", "window": {"start": "admit", "end": "discharge"},
                          "agg": "max"})


def _e06(d):
    d["split"]["key"] = "admission_id"


def _e07(d):
    d["split_unit"] = "family_id"
    d["split"]["key"] = "patient_id"


def _e08(d):
    next(s for s in d["preprocessing"] if s["kind"] == "impute")["fit_scope"] = "all"


def _e10(d):
    d["preprocessing"].append({"name": "univariate_topk", "kind": "select", "k": 5, "fit_scope": "all"})


def _e12(d):
    d["features"].append({"name": "next_admission_unit", "source": "admissions", "column": "unit",
                          "scope": "next_admission", "agg": "value"})


def _e13(d):
    d["features"].append({"name": "renal_orders", "source": "orders",
                          "filter": {"order_type": ["dialysis_order", "nephrology_consult"]},
                          "time_column": "order_time", "window": {"start": "admit", "end": "tp"},
                          "agg": "any"})


def _e14(d):
    d["cohort"]["inclusion"].append({"name": "los_7d", "source": "admissions", "column": "length_of_stay_h",
                                     "agg": "value", "op": ">=", "value": 168})


def _e15(d):
    d["cohort"]["inclusion"].append({"name": "cr_3plus", "source": "labs", "filter": {"test": ["creatinine"]},
                                     "time_column": "report_time",
                                     "window": {"start": "admit", "end": "discharge"},
                                     "agg": "count", "op": ">=", "value": 3})


def _e17(d):
    d["outcome"].pop("ascertainment", None)


PUBLIC_INJECT = {"E01": _e01, "E02": _e02, "E03": _e03, "E06": _e06, "E07": _e07, "E08": _e08,
                 "E10": _e10, "E12": _e12, "E13": _e13, "E14": _e14, "E15": _e15, "E17": _e17}


def inject(case_id: str, kind: str) -> dict:
    d = copy.deepcopy(clean(kind))
    PUBLIC_INJECT[case_id](d)
    d["design_id"] = f"{d['design_id']}+{case_id}"
    return d


# --- 수상해 보이지만 정당한 특징 (동적 AKI 설계에 더함) ---

TRICKY_DYNAMIC = {
    # 이전 입원의 진단 코드: 진단 코드는 퇴원 시각에 알려지고, 이전 입원은 이번 입원 전에 퇴원했다.
    # AKI 코드(N17)가 들어 있어도 다른 입원의 기록이라 결과의 대리 변수가 아니다.
    "prior_admission_dx": {"name": "prior_dx_renal_cardiac", "source": "diagnoses",
                           "scope": "prior_admissions", "filter": {"icd_code": ["N17", "N18", "I50"]},
                           "agg": "any"},
    # 보고 시각으로 거른 검사: tₚ 직전 6시간 안에 보고된 것까지만.
    "report_time_lab_near_tp": {"name": "k_last_6h", "source": "labs", "filter": {"test": ["potassium"]},
                                "time_column": "report_time", "window": {"start": "tp-6h", "end": "tp"},
                                "agg": "last"},
    # 보고 시각으로 거른 이전 입원 검사
    "report_time_lab_prior_admission": {"name": "prior_cr_last", "source": "labs",
                                        "scope": "prior_admissions", "filter": {"test": ["creatinine"]},
                                        "time_column": "report_time", "window": {"end": "tp"},
                                        "agg": "last"},
    # tₚ 이전에 시작된 입원만 센 과거 입원 횟수 (그 환자의 모든 입원 중 입원 시각 ≤ tₚ)
    "admissions_before_tp": {"name": "n_admissions_to_tp", "source": "admissions",
                             "scope": "patient_history", "time_column": "admit_time",
                             "window": {"end": "tp"}, "agg": "count"},
}


def tricky(name: str) -> dict:
    d = copy.deepcopy(clean("동적"))
    d["features"].append(copy.deepcopy(TRICKY_DYNAMIC[name]))
    d["design_id"] = f"{d['design_id']}+{name}"
    return d


def tricky_all() -> dict:
    d = copy.deepcopy(clean("동적"))
    d["features"] += copy.deepcopy(list(TRICKY_DYNAMIC.values()))
    d["design_id"] = "tricky_dynamic_aki"
    return d
