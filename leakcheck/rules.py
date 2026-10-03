"""규칙표 (어댑터). docs/research_plan.md 4절.

사람이 정하고 잠근 규칙을 모든 행에 기계적으로 적용한다. 의미를 해석하지 않는다.

- 1층 구조: 데이터 구조만으로 정해짐 (patient_id 열 → entity, report_time 열 → available_time)
- 2층 맥락: 데이터 밖의 지식. 근거와 함께 적는다 (진단 코드 = 퇴원 시각)
- 3층 불확실: 아무도 확실히 모름. 보수적 기본값 + 가정 기록 (날짜만 = 그날 23:59)

규칙은 "언제 알려지는가"만 말한다. 쓸 수 있는지는 설계의 tₚ가 정한다.
규칙이 없는 테이블·열은 추측하지 않고 RuleMissing으로 멈춘다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RULES_VERSION = "1"

# 진단 코드 코딩 지연(시간). 실제 병원은 퇴원 후 코딩이 늦어 이 값을 키우면
# 고정 시점 재입원 예측에서도 진단 코드가 차단된다. 규칙표 한 줄로 조정한다.
DX_CODING_DELAY_H = 0.0

# Q7 판정 기준: 하위 집단 사이 결과 측정 빈도 비(최대/최소)가 이 값 이상이면 경고.
# 2026-10-02 사용자 확정. 실험 전 고정값이며 이후 바꾸지 않는다.
Q7_RATIO_THRESHOLD = 2.0


class RuleMissing(KeyError):
    """규칙표에 해당 테이블·열의 규칙이 없다. 추측하지 않고 멈춘다."""


@dataclass(frozen=True)
class Avail:
    """값이 알려지는 가장 이른 시각을 정하는 규칙.

    kind
    - "column":     같은 행의 시각 열 `column`이 곧 확인 가능 시각
    - "row_anchor": 그 행이 속한 입원의 기준점(`column` = "admit" | "discharge") + offset_h
    - "date_end":   날짜 열 `column`의 그날 끝(00:00 + 24h - 1분)
    - "static":     입원 전부터 알려진 정적 정보 (입원 시각에 확인 가능으로 본다)

    lag_from: 같은 행의 다른 시각 열 → 확인 가능 시각까지 최대 지연(시간). None = 상한 없음.
    """
    kind: str
    column: str | None
    layer: int
    basis: str
    offset_h: float = 0.0
    lag_from: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class TableRule:
    name: str
    entity_via: str                # "patient_id" 직접 또는 "admission_id"를 거쳐
    id_columns: tuple[str, ...]    # provenance에 쓰는 행 식별 열
    time_columns: dict[str, str]   # 열 → "datetime" | "date"
    value_column: str | None       # 집계할 때 기본 값 열
    row_available: Avail           # 행이 존재한다는 사실(개수 집계 등)이 알려지는 시각
    column_available: dict[str, Avail] = field(default_factory=dict)
    # 이 테이블의 행이 소속 입원의 퇴원 뒤 얼마까지 늦게 알려질 수 있는가(시간). None = 상한 없음
    max_after_discharge_h: float | None = None
    # 결과를 이 테이블로 정하면 측정 빈도가 결과 확인 강도를 좌우한다 (Q7)
    measurement: bool = False
    # 값에서 계산하는 열 (데이터 단계)
    derived: dict[str, str] = field(default_factory=dict)


_ADMIT = Avail("row_anchor", "admit", 1, "입원 사실과 입원 시각은 입원 시각에 기록된다")
_DISCHARGE = Avail("row_anchor", "discharge", 1, "퇴원 시각·퇴원 상태·재원 기간은 퇴원 시각에 확정된다")

TABLES: dict[str, TableRule] = {
    "patients": TableRule(
        name="patients",
        entity_via="patient_id",
        id_columns=("patient_id",),
        time_columns={},
        value_column=None,
        row_available=Avail("static", None, 2, "나이·성별·가족 관계는 입원 전부터 알려진 정적 정보"),
    ),
    "admissions": TableRule(
        name="admissions",
        entity_via="patient_id",
        id_columns=("admission_id",),
        time_columns={"admit_time": "datetime", "discharge_time": "datetime"},
        value_column=None,
        row_available=_ADMIT,
        column_available={
            "admit_time": _ADMIT,
            "unit": Avail("row_anchor", "admit", 2,
                          "입원 병동(ICU/병동)은 입원 시 배정된다. 합성 데이터는 입원당 병동 하나"),
            "discharge_time": _DISCHARGE,
            "discharge_status": _DISCHARGE,
            "length_of_stay_h": _DISCHARGE,
        },
        max_after_discharge_h=0.0,
        derived={"length_of_stay_h": "(discharge_time - admit_time) 시간"},
    ),
    "labs": TableRule(
        name="labs",
        entity_via="admission_id",
        id_columns=("lab_id",),
        time_columns={"collect_time": "datetime", "report_time": "datetime"},
        value_column="value",
        row_available=Avail("column", "report_time", 1,
                            "검사 값은 보고 시각에 알려진다 (채취 시각이 아님)",
                            lag_from={"collect_time": None}),
        max_after_discharge_h=None,  # 퇴원 직전 채취·퇴원 후 보고가 있다
        measurement=True,
    ),
    "diagnoses": TableRule(
        name="diagnoses",
        entity_via="admission_id",
        id_columns=("admission_id", "seq"),
        time_columns={},
        value_column="icd_code",
        row_available=Avail("row_anchor", "discharge", 2,
                            "진단 코드는 퇴원 시 코딩된다 (근거: MIMIC-IV hosp/diagnoses_icd 문서). "
                            "코딩 지연은 DX_CODING_DELAY_H",
                            offset_h=DX_CODING_DELAY_H),
        max_after_discharge_h=DX_CODING_DELAY_H,
    ),
    "procedures": TableRule(
        name="procedures",
        entity_via="admission_id",
        id_columns=("admission_id", "code", "chart_date"),
        time_columns={"chart_date": "date"},
        value_column="code",
        row_available=Avail("date_end", "chart_date", 3,
                            "날짜만 있어 그날 중 언제인지 모른다. 보수적으로 그날 23:59로 가정"),
        max_after_discharge_h=None,
    ),
    "orders": TableRule(
        name="orders",
        entity_via="admission_id",
        id_columns=("order_id",),
        time_columns={"order_time": "datetime"},
        value_column="order_type",
        row_available=Avail("column", "order_time", 1, "오더는 발행 시각에 기록된다"),
        max_after_discharge_h=None,  # 퇴원 후 오더가 없다는 보장은 없다 (3층 보수적)
    ),
}

# 독립 단위의 위계 (작은 것 → 큰 것). 분할 키는 선언된 단위보다 작으면 안 된다.
SPLIT_LEVELS = {"index_row": 0, "admission": 1, "patient": 2, "family": 3}
SPLIT_ALIASES = {
    "index_row": "index_row", "index_id": "index_row", "row": "index_row", "row_id": "index_row",
    "admission": "admission", "admission_id": "admission",
    "patient": "patient", "patient_id": "patient",
    "family": "family", "family_id": "family",
}
SPLIT_KEY_COLUMN = {"index_row": "index_id", "admission": "admission_id",
                    "patient": "patient_id", "family": "family_id"}
# 같은 개체(환자)의 행은 반드시 함께 움직여야 한다. 독립 단위는 이보다 작을 수 없다.
ENTITY_LEVEL = "patient"
DEFAULT_SPLIT_UNIT = "patient"

# 데이터에서 무엇을 추정하는 단계가 허용되는 적합 범위
FIT_SCOPES_OK = {"train", "train_fold"}

# 결과 정의별 대리 변수 (2층: 결과를 정하는 임상 과정에서 생기는 기록). 설계서의 proxies와 합쳐 쓴다.
OUTCOME_PROXIES: dict[str, list[dict]] = {
    "kdigo_creatinine": [
        {"source": "orders", "filter": {"order_type": ["dialysis_order", "nephrology_consult",
                                                       "renal_ultrasound"]},
         "basis": "AKI를 의심·확인한 뒤 내는 오더"},
        {"source": "procedures", "filter": {"code": ["5A1D70Z"]}, "basis": "투석 처치"},
        {"source": "diagnoses", "filter": {"icd_code": ["N17"]}, "basis": "AKI 진단 코드"},
    ],
    "next_admission": [],
}

# 결과 확인 강도가 다를 수 있는 층 (2층: ICU는 병동보다 검사를 자주 한다).
# 데이터가 있으면 ASCERTAINMENT_CANDIDATES 전체에서 측정 빈도 비를 계산한다.
ASCERTAINMENT_STRATA = [("admissions", "unit")]
ASCERTAINMENT_CANDIDATES = [("admissions", "unit"), ("patients", "sex")]


# 앞부분 일치로 거르는 코드 열 (N17 → N17.9)
CODE_COLUMNS = {"icd_code", "code"}


def default_time_column(table: str) -> str | None:
    """창을 적용할 기본 시각 열: 확인 가능 시각 열 (입원은 입원 시각, 진단은 없음)."""
    a = table_rule(table).row_available
    if a.kind in ("column", "date_end"):
        return a.column
    if table == "admissions":
        return "admit_time"
    return None


def table_rule(table: str) -> TableRule:
    try:
        return TABLES[table]
    except KeyError:
        raise RuleMissing(f"규칙표에 테이블 '{table}'의 규칙이 없다") from None


def availability(table: str, column: str | None = None) -> Avail:
    """(테이블, 열) 값이 알려지는 시각의 규칙. 열 규칙이 없으면 행 규칙을 쓴다."""
    rule = table_rule(table)
    if column is not None and column in rule.column_available:
        return rule.column_available[column]
    return rule.row_available


def split_level(name: str) -> str:
    try:
        return SPLIT_ALIASES[name]
    except KeyError:
        raise RuleMissing(f"분할 단위·키 '{name}'가 규칙표의 위계에 없다") from None
