"""채점기 (5단계에 고정, `experiment/scoring_rules.md`를 코드로 옮긴 것).

보고서 → (지목 항목, 문제 종류) 목록 추출 → 정답표와 대조 → 탐지 · 미탐지 · 오경보.
판정은 모두 규칙 기반이다. LLM은 쓰지 않는다 (CLAUDE.md 절대 규칙 7).

맹검: 채점 단계(`grade_*`)는 보고서 id · 변형 파일 · 본문만 받는다. 조건(가/나) 표시는
채점이 끝난 뒤 `summarize`에서만 붙인다. 조건 칸이 섞인 기록은 채점하지 않고 멈춘다.

사용 (7단계):
    python -m experiment.grader grade runs/reports --checker runs/checker --out results/grades.json
    python -m experiment.grader summarize results/grades.json runs/conditions.json --out results/summary.json
암호는 tools.seal과 같은 방식(환경 변수 SEAL_PASSWORD 또는 터미널 입력)으로만 받는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VARIANT_DIR = ROOT / "designs" / "variants"
ANSWER_KEY = ROOT / "sealed" / "answer_key.enc"
INJECTION_LOG = ROOT / "docs" / "injection_log.md"

KINDS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q7"]   # 채점하는 문제 종류 (Q6는 기록만이라 채점하지 않음)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261003
CONDITION_FIELDS = {"condition", "arm", "조건", "skill"}

# ---------------------------------------------------------------- 문제 종류 핵심어 사전
# 문제 설명(problem)을 소문자로 바꾼 뒤 부분 문자열로 찾는다. 한 설명이 여러 종류에 걸릴 수 있다.
# 점검기 표현과 일반 표현을 한국어·영어로 함께 넣었다. 실험 전에 고정하며 결과를 본 뒤 바꾸지 않는다.

KEYWORDS: dict[str, list[str]] = {
    "Q1": [  # 가용 시점
        "가용 시점", "확인 가능 시각", "확인 가능한 시각", "알려진 시각", "알려지는 시각",
        "예측 시점 이후", "예측 시점 뒤", "예측 시점보다 늦", "예측 시점에는 알", "예측 시점에 알",
        "예측 시점에 아직", "예측 시점에서 알", "예측 시각 이후", "예측 시각 뒤",
        "tp 이후", "tp 뒤", "tp보다 늦", "tp 시점에는 알", "tp에는 알", "tp에 아직", "tp에 알",
        "기준 시점(", "뒤일 수 있다", "데이터에서 늦게 알려진",
        "미래 정보", "미래의 정보", "미래 데이터", "미래 시점", "아직 알려지지", "아직 알 수 없",
        "아직 나오지", "아직 보고되지", "보고 시각", "보고되기 전", "보고 지연",
        "채취 시각", "채취 시점", "퇴원 시점에야", "퇴원 후에야", "퇴원 때 확정", "퇴원 시 확정",
        "입원 전체 기간", "입원 전체의", "입원 기간 전체", "시간적 누수", "시점 누수", "시간 누수",
        "look-ahead", "lookahead", "look ahead", "future information", "future data", "future value",
        "information from the future", "leakage from the future", "not available at prediction",
        "not yet available", "unavailable at prediction", "not available at the time",
        "available only after", "only known after", "known only at", "after the prediction time",
        "after prediction time", "after the prediction point", "after the landmark",
        "temporal leakage", "time leakage", "report time", "reported after", "result time",
        "collection time", "collect_time", "whole admission", "entire admission", "entire stay",
        "퇴원 후에 코딩", "퇴원 후 코딩", "퇴원 뒤에 코딩", "코딩 지연", "코딩되어",
        "예측 시점에 사용할 수 없", "예측 시점에는 사용할 수 없", "예측 시점에 쓸 수 없", "예측 시점에는 쓸 수 없",
        "예측 시점에 이용할 수 없", "예측 시점에 존재하지", "예측 시점에는 존재하지",
        "coded after discharge", "coding delay", "not usable at prediction", "cannot be used at prediction",
    ],
    "Q2": [  # 독립 단위
        "독립 단위", "분할 단위", "분할 키", "분할 묶음", "그룹 분할", "묶음 분할",
        "같은 환자", "동일 환자", "같은 가족", "동일 가족", "가족 단위", "환자 단위",
        "학습과 평가에 함께", "학습·평가에", "학습/평가에", "학습과 평가 양쪽", "학습과 평가에 나뉘",
        "학습과 테스트", "학습 세트와 테스트", "서로 독립이 아", "독립적이지 않", "독립이 아니",
        "행 단위 분할", "행 단위로 분할", "입원 단위 분할", "입원 단위로 분할", "무작위 행 분할",
        "교차검증 키", "교차 검증 키", "교차검증 묶음",
        "same patient", "same family", "patient-level split", "patient level split", "family-level",
        "group split", "grouped split", "group k-fold", "groupkfold", "group-aware",
        "independent unit", "unit of independence", "not independent", "non-independent",
        "split key", "split unit", "split by row", "row-level split", "admission-level split",
        "both train and test", "both training and test", "train and test sets", "across splits",
        "cv key", "cross-validation key", "patient overlap", "subject overlap",
        "독립성이 깨", "독립성을 깨", "독립성 위반", "독립성을 위반", "독립성을 해", "독립성이 없",
        "학습과 검증에", "학습과 검증 세트", "학습·검증", "학습/검증", "가족 구성원이",
        "행 단위로 나누", "행 단위로 분할", "한 사람의 데이터", "같은 사람", "동일 인물", "양쪽에 섞", "양쪽에 들어",
        "independence", "same person", "same individual", "both sides of the split",
    ],
    "Q3": [  # 적합 범위
        "적합 범위", "fit_scope", "fit scope", "범위에서 적합",
        "전체 데이터로 적합", "전체 데이터에서 적합", "전체 데이터로 추정", "전체 데이터에서 추정",
        "전체 데이터를 사용해 추정", "전체 데이터로 계산", "전체 데이터에서 계산", "전체 데이터로 학습",
        "전체 데이터에서 선택", "전체 데이터로 선택", "분할 전에", "분할 이전에", "분할하기 전",
        "평가 부분의 정보", "평가 데이터의 정보", "테스트 데이터의 정보", "평가 데이터 정보",
        "학습 부분에서만", "학습 데이터에서만", "전처리 누수", "특징 선택 누수", "변수 선택 누수",
        "fit on the full", "fit on all", "fitted on all", "fit on the entire", "fitted on the entire",
        "fit on the whole", "fitted on the whole", "entire dataset", "whole dataset", "full dataset",
        "before splitting", "before the split", "prior to splitting", "test set information",
        "information from the test", "training data only", "training set only", "train only",
        "train_fold", "preprocessing leakage", "feature selection leakage", "data snooping",
        "전체 표본에서", "전체 표본으로", "전체 표본을", "전체 자료에서", "전체 자료로",
        "시험 세트 정보", "시험 세트의 정보", "검증 세트 정보", "검증 세트의 정보", "테스트 세트 정보",
        "테스트 세트의 정보", "평가 세트 정보", "평가 세트의 정보", "미리 본", "엿보",
        "peek", "test data leak into", "whole sample", "full sample", "entire sample",
    ],
    "Q4": [  # 결과 출처
        "결과 출처", "결과 정의", "결과를 정한", "결과를 정의", "결과를 정하는", "결과와 같은 행",
        "결과 창", "결과 기간", "결과와 겹", "대리 변수", "대리변수", "결과의 대리", "결과를 반영",
        "결과 정보", "결과가 섞", "결과 누수", "레이블 누수", "라벨 누수", "타깃 누수", "타겟 누수",
        "결과 자체", "결과를 직접", "결과를 암시", "결과를 그대로",
        "target leakage", "label leakage", "outcome leakage", "proxy", "outcome definition",
        "defines the outcome", "define the outcome", "outcome window", "encodes the outcome",
        "reflects the outcome", "same row as the outcome", "derived from the outcome",
        "consequence of the outcome", "downstream of the outcome",
        "예측하려는 결과", "예측할 결과", "예측하는 결과", "예측 대상인 결과", "예측 대상 자체", "결과 그 자체",
        "outcome we are predicting", "outcome being predicted", "is the outcome itself", "is itself the outcome",
    ],
    "Q5": [  # 선택 시점
        "선택 시점", "선택 편향", "불멸 시간", "생존 편향", "코호트 선택 시점", "대상자 선택 시점",
        "기준 시점 이후", "기준 시점 뒤", "immortal time", "selection bias", "survivor bias",
        "survivorship", "selection time", "conditioning on the future",
        "conditioned on the future",
    ],
    "Q7": [  # 결과 확인 균질성
        "확인 균질성", "확인 강도", "측정 강도", "측정 빈도", "측정 횟수가 다", "검사 빈도",
        "검사 횟수가 다", "감시 편향", "탐지 편향", "확인 편향", "결과 확인", "층별로 다루",
        "ascertainment", "surveillance bias", "detection bias", "verification bias",
        "measurement frequency", "testing frequency", "monitoring intensity", "measured more often",
        "tested more often", "differential measurement", "differential testing", "informative observation",
    ],
    "Q6": [  # 다중 시도 (인식만 하고 채점하지 않음)
        "다중 시도", "다중 비교", "다중 검정", "시도 횟수", "multiple testing", "multiple comparison",
        "number of attempts", "p-hacking", "forking paths",
    ],
}

UNABLE = [  # 점검 불가 표현: 문제 종류 핵심어보다 먼저 본다
    "점검 불가", "확인 불가", "판정 불가", "규칙표에 없", "규칙이 없어", "규칙이 없다", "출처 불명",
    "cannot be checked", "could not be checked", "could not check", "unable to check", "not checkable",
    "no rule for", "unknown provenance",
]

_LABEL = re.compile(r"(?<![a-z0-9])q([1-7])(?![0-9])")
_RANGE = re.compile(r"(?<![a-z0-9])q([1-7])\s*[~\-–]\s*q([1-7])(?![0-9])")


@dataclass
class Kind:
    status: str                       # "kinds" | "unknown"(종류 불명) | "unable"(점검 불가) | "q6"
    qs: frozenset = frozenset()


def classify(problem: str, item: str | None = None) -> Kind:
    """문제 설명 → 문제 종류. item이 포함·제외 기준이면 시점 문제(Q1)를 선택 시점(Q5)으로도 본다."""
    t = problem.lower().replace("tₚ", "tp").replace("t_p", "tp")
    if any(u in t for u in UNABLE):
        return Kind("unable")
    qs = {f"Q{m}" for m in _LABEL.findall(t)}
    for a, b in _RANGE.findall(t):
        qs |= {f"Q{i}" for i in range(int(a), int(b) + 1)}
    qs |= {q for q, words in KEYWORDS.items() if any(w in t for w in words)}
    if item and item.startswith("cohort.") and "Q1" in qs:
        qs.add("Q5")
    if qs == {"Q6"}:
        return Kind("q6")
    qs.discard("Q6")
    return Kind("kinds", frozenset(qs)) if qs else Kind("unknown")


# ---------------------------------------------------------------- 설계서 항목

LIST_SECTIONS = {"features": "features", "inclusion": "cohort.inclusion",
                 "exclusion": "cohort.exclusion", "preprocessing": "preprocessing"}
FIXED_ITEMS = ["split", "outcome", "tp", "cohort.subgroups", "cohort.index_time"]
NOT_GRADED = {"model", "attempts"}   # 지목해도 오경보로 세지 않는 항목 (model.family 등, Q6 시도 기록)


def _names(design: dict) -> dict[str, list[str]]:
    c = design.get("cohort", {})
    return {"features": [f["name"] for f in design.get("features", [])],
            "inclusion": [x["name"] for x in c.get("inclusion", [])],
            "exclusion": [x["name"] for x in c.get("exclusion", [])],
            "preprocessing": [p["name"] for p in design.get("preprocessing", [])]}


def universe(design: dict) -> list[str]:
    """채점 대상 항목 전체 (H2 무작위 기대치의 모집단)."""
    n = _names(design)
    return ([f"{LIST_SECTIONS[s]}.{x}" for s in LIST_SECTIONS for x in n[s]] + FIXED_ITEMS)


def _parts(s: str) -> list[str]:
    s = s.strip().strip("`'\" ")
    s = re.sub(r"\[\s*(?:name\s*=\s*)?['\"]?([^\]'\"]+?)['\"]?\s*\]", r".\1", s)
    s = s.replace(":", ".").replace("/", ".")
    return [p.strip() for p in s.split(".") if p.strip()]


def _canon_parts(p: list[str], design: dict) -> str | None:
    if not p:
        return None
    n = _names(design)
    head, rest = p[0], p[1:]
    if head == "cohort":
        if not rest:
            return None
        head, rest = rest[0], rest[1:]
        if head in ("subgroups", "index_time"):
            return f"cohort.{head}"
        if head not in ("inclusion", "exclusion"):
            return None
    if head in LIST_SECTIONS:
        return f"{LIST_SECTIONS[head]}.{rest[0]}" if rest and rest[0] in n[head] else None
    if head in ("split", "split_unit", "cv_key"):
        return "split"
    if head == "model":
        return "split" if rest and rest[0] == "tuning" else "model"
    if head in ("outcome", "ascertainment"):
        if head == "ascertainment":
            rest = ["ascertainment"]
        if rest and rest[0] == design.get("outcome", {}).get("name"):
            rest = rest[1:]
        return ".".join(["outcome"] + rest)
    if head == "tp":
        return ".".join(["tp"] + rest)
    if head in ("subgroups", "index_time"):
        return f"cohort.{head}"
    if head == "attempts":
        return "attempts"
    hits = [s for s in LIST_SECTIONS if head in n[s]]
    if len(hits) == 1:
        return f"{LIST_SECTIONS[hits[0]]}.{head}"
    return None


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|:)[A-Za-z0-9_]+|\[[^\]]*\])*")


def resolve(target: str, design: dict) -> str | None:
    """보고서의 target 문자열 → 설계서 항목(정규형). 정할 수 없으면 None(대상 불명)."""
    whole = _canon_parts(_parts(target), design)
    if whole is not None:
        return whole
    toks = _TOKEN.findall(target)
    qualified = [t for t in toks if re.search(r"[.:\[]", t)]
    for group in (qualified, toks):   # 점 경로로 적은 것을 먼저 본다
        found = {c for tok in group if (c := _canon_parts(_parts(tok), design)) is not None}
        if found:
            return found.pop() if len(found) == 1 else None
    return None


def _prefix(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def key_target(t: str, design: dict) -> str:
    """정답표의 targets 칸(inject.targets_of 형식) → 정규형. 'cohort'·'model'은 묶음 그대로 둔다."""
    p = _parts(t)
    if p == ["cohort"] or p == ["model"]:
        return p[0]
    return _canon_parts(p, design) or ".".join(p)


def matches(item: str, key: str) -> bool:
    """보고서 항목이 정답표 항목을 가리키는가 (scoring_rules.md 2절 동의어 표)."""
    if key == "cohort":
        return item.startswith("cohort.")
    if key == "model":
        return item in ("split", "model")
    if key.split(".")[0] in ("outcome", "tp"):
        # 결과 확인 방식이 정답이면 그 층을 정하는 하위 집단 설정도 같은 항목으로 본다
        if item == "cohort.subgroups" and (key + ".").startswith("outcome.ascertainment."):
            return True
        return _prefix(item, key)
    return item == key


def top_item(item: str) -> str:
    """오경보 중복 제거·H2 모집단 비교용 최상위 항목 (outcome.*, tp.* → outcome, tp)."""
    head = item.split(".")[0]
    return head if head in ("outcome", "tp") else item


def graded_item(item: str) -> bool:
    return item not in NOT_GRADED


# ---------------------------------------------------------------- 보고서 목록 추출

_BLOCK = re.compile(r"```findings[^\n]*\n(.*?)```", re.S)


@dataclass
class Entry:
    target: str
    problem: str
    item: str | None = None
    kind: Kind = field(default_factory=lambda: Kind("unknown"))


def parse_findings(text: str) -> tuple[list[dict], str | None, int]:
    """보고서 본문의 마지막 ```findings 블록 → (항목 목록, 형식 오류, 형식이 틀린 항목 수)."""
    blocks = _BLOCK.findall(text)
    if not blocks:
        return [], "목록 없음", 0
    try:
        data = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return [], "JSON 오류", 0
    if not isinstance(data, list):
        return [], "목록 아님", 0
    out, bad = [], 0
    for x in data:
        if not isinstance(x, dict) or not isinstance(x.get("problem"), str):
            bad += 1
            continue
        tg = x.get("target")
        tgs = [tg] if isinstance(tg, str) else tg if isinstance(tg, list) else None
        if not tgs or not all(isinstance(s, str) for s in tgs):
            bad += 1
            continue
        out += [{"target": s, "problem": x["problem"]} for s in tgs]
    return out, None, bad


# ---------------------------------------------------------------- 채점

@dataclass
class Grade:
    report_id: str
    variant: str
    source: str                                  # "report" | "checker"
    format_error: str | None = None
    n_entries: int = 0
    n_malformed: int = 0
    n_unable: int = 0                            # 점검 불가 지적 (탐지도 오경보도 아님)
    n_unknown_kind: int = 0                      # 종류 불명 지적 (주 분석은 항목으로 판정, 보조 분석 미탐지)
    n_unresolved: int = 0                        # 대상 불명 지적 (설계서 항목으로 정할 수 없음)
    n_q6: int = 0                                # 다중 시도 지적 (채점하지 않음)
    defects: list[dict] = field(default_factory=list)
    false_alarms: list[str] = field(default_factory=list)
    flagged_items: list[str] = field(default_factory=list)  # 지목한 채점 대상 항목 (최상위, 중복 없음)

    def to_dict(self) -> dict:
        return asdict(self)


def _check_blind(rec: dict) -> None:
    bad = CONDITION_FIELDS & set(rec)
    if bad:
        raise ValueError(f"채점 입력에 조건 칸 {sorted(bad)}이 있다. 조건은 채점 뒤에만 붙인다.")


def _score(g: Grade, entries: list[Entry], design: dict, key_entry: dict) -> Grade:
    keys = [(d, [key_target(t, design) for t in d["targets"]]) for d in key_entry["defects"]]
    pointing = []
    for e in entries:
        if e.kind.status == "unable":
            g.n_unable += 1
        elif e.kind.status == "q6":
            g.n_q6 += 1
        elif e.item is None:
            g.n_unresolved += 1
        else:
            if e.kind.status == "unknown":
                g.n_unknown_kind += 1
            pointing.append(e)
    for d, kt in keys:
        hit = [e for e in pointing if any(matches(e.item, k) for k in kt)]
        qs = set().union(*(e.kind.qs for e in hit)) if hit else set()
        g.defects.append({"id": d["id"], "question": d["question"], "holdout": d["holdout"],
                          "adjusted": bool(d.get("adjustment")), "targets": kt,
                          "primary": bool(hit), "secondary": d["question"] in qs,
                          "extra_kinds": sorted(qs - {d["question"]})})
    fa, flagged = [], []
    for e in pointing:
        if not graded_item(e.item):
            continue
        if top_item(e.item) not in flagged:
            flagged.append(top_item(e.item))
        if not any(matches(e.item, k) for _, kt in keys for k in kt) and e.item not in fa:
            fa.append(e.item)
    g.false_alarms, g.flagged_items = fa, flagged
    return g


def grade_report(rec: dict, design: dict, key_entry: dict) -> Grade:
    """에이전트 보고서 1개. rec = {"report_id", "variant", "text"} (조건 칸 없음)."""
    _check_blind(rec)
    raw, err, bad = parse_findings(rec["text"])
    g = Grade(rec["report_id"], rec["variant"], "report", err, len(raw), bad)
    entries = []
    for x in raw:
        item = resolve(x["target"], design)
        entries.append(Entry(x["target"], x["problem"], item, classify(x["problem"], item)))
    return _score(g, entries, design, key_entry)


def grade_checker(rec: dict, design: dict, key_entry: dict) -> Grade:
    """(가) 조건의 점검기 출력(JSON) 1개. rec = {"report_id", "variant", "exit_code", "checker": Report.to_dict() 또는 None}.
    점검기만으로 본 탐지율(2차 지표)에 쓴다. 문제 종류는 점검기의 question 칸을 그대로 쓴다."""
    _check_blind(rec)
    g = Grade(rec["report_id"], rec["variant"], "checker")
    out = rec.get("checker")
    if rec.get("exit_code") == 2 or out is None:
        g.format_error, g.n_unable = "점검 불가(종료 코드 2)", 1
        return _score(g, [], design, key_entry)
    entries = []
    for f in out["findings"]:
        if f["verdict"] not in ("차단", "경고"):
            continue
        q, reason = f["question"], f["reason"]
        if q not in KINDS + ["Q6"] or reason.startswith("출처 불명") or "확인 불가" in reason:
            kind = Kind("unable")
        elif q == "Q6":
            kind = Kind("q6")
        else:
            kind = Kind("kinds", frozenset({q}))
        entries.append(Entry(f["target"], reason, resolve(f["target"], design), kind))
    g.n_entries = len(entries)
    return _score(g, entries, design, key_entry)


# ---------------------------------------------------------------- 정답표 · 변형 파일

def load_answer_key(password: str, path: Path = ANSWER_KEY) -> dict:
    """7단계 전용. 정답표를 메모리에서만 복호화한다 (평문을 파일로 쓰지 않는다)."""
    from tools.seal import decrypt_bytes
    return json.loads(decrypt_bytes(path.read_bytes(), password))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def injection_log_hashes(path: Path = INJECTION_LOG) -> dict[str, str]:
    rows = re.findall(r"^\| designs/variants/(design_\w+\.json) \| `([0-9a-f]{64})` \|", path.read_text(encoding="utf-8"), re.M)
    return dict(rows)


def verify_variants(key_variants: dict, variant_dir: Path = VARIANT_DIR, log: dict[str, str] | None = None) -> list[str]:
    """변형 파일 SHA-256을 정답표·주입 기록과 대조. 어긋난 항목 목록(빈 목록이면 일치)."""
    log = injection_log_hashes() if log is None else log
    bad = []
    files = {p.name for p in variant_dir.glob("design_*.json")}
    if files != set(key_variants):
        bad.append(f"파일 목록 불일치: {sorted(files ^ set(key_variants))}")
    for fname, entry in key_variants.items():
        if fname not in files:
            continue
        h = sha256_file(variant_dir / fname)
        if h != entry["sha256"]:
            bad.append(f"{fname}: 정답표 해시와 다름")
        if log.get(fname) != h:
            bad.append(f"{fname}: injection_log.md 해시와 다름")
    return bad


def load_design(fname: str, variant_dir: Path = VARIANT_DIR) -> dict:
    return json.loads((variant_dir / fname).read_text(encoding="utf-8"))


def grade_all(reports: list[dict], key_variants: dict, checker: list[dict] = (),
              designs: dict[str, dict] | None = None) -> list[dict]:
    """맹검 채점. 보고서·점검기 기록 → Grade 사전 목록 (조건 없음)."""
    designs = designs if designs is not None else {}
    out = []
    for rec, fn in [(r, grade_report) for r in reports] + [(c, grade_checker) for c in checker]:
        v = rec["variant"]
        if v not in designs:
            designs[v] = load_design(v)
        out.append(fn(rec, designs[v], key_variants[v]).to_dict())
    return out


# ---------------------------------------------------------------- 요약 · H1 · H2

def _rate(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _placements(grades: list[dict], keep=lambda d: True) -> list[dict]:
    return [{**d, "variant": g["variant"], "report_id": g["report_id"]}
            for g in grades for d in g["defects"] if keep(d)]


def detection(grades: list[dict], keep=lambda d: True) -> dict:
    p = _placements(grades, keep)
    by_q = {q: {"n": len(x), "primary": _rate([d["primary"] for d in x]), "secondary": _rate([d["secondary"] for d in x])}
            for q in KINDS if (x := [d for d in p if d["question"] == q])}
    return {"n": len(p), "primary": _rate([d["primary"] for d in p]),
            "secondary": _rate([d["secondary"] for d in p]), "by_question": by_q}


def false_alarms(grades: list[dict], key_variants: dict) -> dict:
    out = {}
    for label, clean in (("clean", True), ("defect", False)):
        gs = [g for g in grades if (not key_variants[g["variant"]]["defects"]) == clean]
        n = [len(g["false_alarms"]) for g in gs]
        out[label] = {"reports": len(gs), "total": sum(n), "mean_per_report": (sum(n) / len(n)) if n else None}
    return out


def bootstrap_diff(ga: list[dict], gb: list[dict], measure: str = "primary",
                   n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """탐지율 차이 (가 − 나), 결함 변형 단위 짝지은 부트스트랩 백분위 95% CI."""
    variants = sorted({g["variant"] for g in ga + gb if g["defects"]})

    def counts(gs):
        det = np.array([sum(d[measure] for g in gs if g["variant"] == v for d in g["defects"]) for v in variants], float)
        tot = np.array([sum(len(g["defects"]) for g in gs if g["variant"] == v) for v in variants], float)
        return det, tot

    (da, ta), (db, tb) = counts(ga), counts(gb)
    if not variants or ta.sum() == 0 or tb.sum() == 0:
        return {"diff": None, "ci95": None, "n_variants": len(variants)}
    est = da.sum() / ta.sum() - db.sum() / tb.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(variants), size=(n, len(variants)))
    sa, sta, sb, stb = da[idx].sum(1), ta[idx].sum(1), db[idx].sum(1), tb[idx].sum(1)
    ok = (sta > 0) & (stb > 0)
    diffs = sa[ok] / sta[ok] - sb[ok] / stb[ok]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(est), "ci95": [float(lo), float(hi)], "n_variants": len(variants),
            "n_boot": int(ok.sum()), "seed": seed}


def random_hit_prob(n_items: int, k_match: int, m_flags: int) -> tuple[float, float]:
    """m개 항목을 무작위로 골라(비복원) 각각에 종류 6개 중 하나를 무작위로 붙였을 때
    k개 정답 항목 중 하나라도 맞힐 확률 → (주 분석: 항목만, 보조 분석: 항목+종류)."""
    m = min(m_flags, n_items)
    if m == 0 or k_match == 0:
        return 0.0, 0.0
    total = math.comb(n_items, m)
    item_only = 1 - math.comb(n_items - k_match, m) / total
    item_kind = sum(math.comb(k_match, j) * math.comb(n_items - k_match, m - j) / total * (1 - (5 / 6) ** j)
              for j in range(1, min(k_match, m) + 1))
    return item_only, item_kind


def _poisson_binomial_sf(ps: list[float], k: int) -> float:
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def h2(grades: list[dict], designs: dict[str, dict], keep=lambda d: d["holdout"]) -> dict:
    """보류 사례 탐지율 vs 같은 수의 지적을 무작위 배치했을 때의 기대 탐지율 (보고서별 지적 수 그대로)."""
    rows = []
    for g in grades:
        design = designs[g["variant"]]
        uni = universe(design)
        m = len([i for i in g["flagged_items"] if i in uni])
        for d in g["defects"]:
            if not keep(d):
                continue
            k = sum(any(matches(u, t) for t in d["targets"]) for u in uni)
            pri, sec = random_hit_prob(len(uni), k, m)
            rows.append({"primary": d["primary"], "secondary": d["secondary"], "p_pri": pri, "p_sec": sec})
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for meas, pk in (("primary", "p_pri"), ("secondary", "p_sec")):
        obs = sum(r[meas] for r in rows)
        exp = sum(r[pk] for r in rows) / len(rows)
        out[meas] = {"observed": obs / len(rows), "expected_random": exp,
                     "met": obs / len(rows) > exp,
                     "p_one_sided": _poisson_binomial_sf([r[pk] for r in rows], obs)}
    return out


def stability(grades: list[dict], measure: str = "primary") -> float | None:
    """같은 (변형, 결함)에 대한 반복 실행들이 모두 같은 판정인 비율."""
    groups: dict[tuple, list[bool]] = {}
    for d in _placements(grades):
        groups.setdefault((d["variant"], d["id"]), []).append(d[measure])
    multi = [v for v in groups.values() if len(v) > 1]
    return _rate([len(set(v)) == 1 for v in multi])


def _counts(gs: list[dict]) -> dict:
    return {k: sum(g[k] for g in gs) for k in ("n_unknown_kind", "n_unable", "n_unresolved", "n_q6", "n_malformed")} | {
        "format_errors": sum(g["format_error"] is not None for g in gs),
        "extra_kinds_on_defects": sum(len(d["extra_kinds"]) > 0 for g in gs for d in g["defects"])}


def summarize(grades: list[dict], conditions: dict[str, dict], key_variants: dict,
              designs: dict[str, dict] | None = None) -> dict:
    """채점 결과에 조건을 붙여 지표·H1·H2를 계산한다 (docs/success_criteria.md)."""
    designs = designs if designs is not None else {}
    for v in key_variants:
        if v not in designs:
            designs[v] = load_design(v)
    reps = [g for g in grades if g["source"] == "report"]
    chk = [g for g in grades if g["source"] == "checker"]
    by = {c: [g for g in reps if conditions[g["report_id"]]["condition"] == c] for c in ("가", "나")}
    unadj = lambda d: d["holdout"] and not d["adjusted"]
    out: dict = {"conditions": {}}
    for c, gs in by.items():
        out["conditions"][c] = {
            "reports": len(gs),
            "detection": {"all": detection(gs), "public": detection(gs, lambda d: not d["holdout"]),
                          "holdout": detection(gs, lambda d: d["holdout"]),
                          "holdout_unadjusted": detection(gs, unadj)},
            "false_alarms": false_alarms(gs, key_variants),
            "counts": _counts(gs),
            "stability": {"primary": stability(gs, "primary"), "secondary": stability(gs, "secondary")},
        }
    ck = [g for g in chk if conditions[g["report_id"]]["condition"] == "가"]
    out["checker_only"] = {"runs": len(ck), "detection": detection(ck),
                           "holdout": detection(ck, lambda d: d["holdout"]),
                           "false_alarms": false_alarms(ck, key_variants), "counts": _counts(ck)}
    fa_clean = out["conditions"]["가"]["false_alarms"]["clean"]["mean_per_report"]
    h1 = {}
    for meas in ("primary", "secondary"):
        b = bootstrap_diff(by["가"], by["나"], meas)
        h1[meas] = {**b, "fa_clean_mean_ga": fa_clean,
                    "met": b["ci95"] is not None and b["ci95"][0] > 0 and fa_clean is not None and fa_clean <= 1}
    out["H1"] = h1
    out["H2"] = {"all_holdout": {"가": h2(by["가"], designs), "나": h2(by["나"], designs)},
                 "unadjusted_holdout": {"가": h2(by["가"], designs, unadj), "나": h2(by["나"], designs, unadj)}}
    disagree = [name for name, a, b in [
        ("H1", h1["primary"]["met"], h1["secondary"]["met"]),
        ("H2", out["H2"]["all_holdout"]["가"].get("primary", {}).get("met"),
         out["H2"]["all_holdout"]["가"].get("secondary", {}).get("met"))] if a != b]
    out["primary_secondary_disagree"] = disagree
    return out


# ---------------------------------------------------------------- 명령행

def _read_dir(d: Path | None) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade", help="맹검 채점 (조건 파일을 읽지 않는다)")
    g.add_argument("reports", type=Path)
    g.add_argument("--checker", type=Path)
    g.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize", help="채점 결과 + 조건 → 지표·H1·H2")
    s.add_argument("grades", type=Path)
    s.add_argument("conditions", type=Path)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    from tools.seal import get_password
    key = load_answer_key(get_password())
    bad = verify_variants(key["variants"])
    if bad:
        raise SystemExit("변형 파일 해시 불일치:\n" + "\n".join(bad))
    if a.cmd == "grade":
        grades = grade_all(_read_dir(a.reports), key["variants"], _read_dir(a.checker))
        a.out.write_text(json.dumps(grades, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        grades = json.loads(a.grades.read_text(encoding="utf-8"))
        conds = json.loads(a.conditions.read_text(encoding="utf-8"))
        summary = summarize(grades, conds, key["variants"])
        a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
