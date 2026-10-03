"""결함 주입과 맹검 변형 (4단계).

깨끗한 기본 설계서(designs/base/)에 오류 사례를 "패치"로 심어 유형별 변형 10개(결함 8 + 깨끗한 것 2)를 만든다.
- 패치: 설계서의 어느 칸을 어떻게 바꾸는지 적은 명령 목록 (apply_ops)
- 공개 12개 사례의 패치는 이 파일에 있다 (이미 공개된 내용).
- 보류 6개 사례의 패치는 이 파일에 없다. 실행할 때 표준 입력(JSON)으로만 받고,
  재현용으로 정답표 봉인본(sealed/answer_key.enc) 안에만 넣는다.
- 난수 시드도 정답표 봉인본에만 넣는다 (시드 + 공개 패치로 배치를 다시 계산할 수 없게).

맹검 처리
- 파일명과 design_id는 무작위 코드 (design_XXXX.json). 사례 id는 변형 어디에도 없다.
- 심은 항목의 이름은 기본 설계의 다른 항목과 같은 방식으로 붙인다.
- 특징·포함 기준 순서는 20개 변형 모두에서 섞는다 (덧붙인 항목이 맨 뒤에 있지 않게).
- 유형별 깨끗한 변형 2개 중 1개에는 "수상해 보이지만 정당한" 변경을 넣는다 (정답표에 결함 아님으로 기록).

사용 예 (보류 패치는 표준 입력으로):
    SEAL_PASSWORD=... python -m designs.inject < 보류패치.json
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import random
import re
import secrets
import sys
from collections import Counter
from pathlib import Path

from tools.seal import decrypt_bytes, encrypt_bytes, get_password

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "designs" / "base"
VARIANT_DIR = ROOT / "designs" / "variants"
ANSWER_KEY = ROOT / "sealed" / "answer_key.enc"
BASE_FILES = {"dynamic": "dynamic_aki.json", "fixed": "fixed_readmission.json"}
TYPE_KO = {"동적": ["dynamic"], "고정": ["fixed"], "둘다": ["dynamic", "fixed"]}

N_DEFECT_VARIANTS = 8      # 유형별 결함 변형 수
N_CLEAN_VARIANTS = 2       # 유형별 깨끗한 변형 수 (그중 1개는 정당한 변경)
MAX_DEFECTS = 2            # 변형당 결함 수 1~2
SOLO_CASES = {"E16"}       # 다른 결함과 같은 변형에 넣지 않는 사례 (2026-10-03 사용자 결정)


# --- 패치 적용 -------------------------------------------------------------

_STEP = re.compile(r"^(\w+)(?:\[name=([^\]]+)\])?$")


def _walk(d: dict, path: str):
    """'a.b[name=x].c' → (부모 객체, 마지막 키)."""
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        key, name = _STEP.match(part).groups()
        cur = cur[key]
        if name is not None:
            matches = [x for x in cur if x.get("name") == name]
            if len(matches) != 1:
                raise KeyError(f"{path}: name={name} 항목이 {len(matches)}개")
            cur = matches[0]
    return cur, parts[-1]


def apply_ops(design: dict, ops: list[dict]) -> dict:
    """패치 명령을 적용한 복사본. 명령: set, append, insert_after, remove."""
    d = copy.deepcopy(design)
    for op in ops:
        parent, key = _walk(d, op["path"])
        kind = op["op"]
        if kind == "set":
            if key not in parent:
                raise KeyError(f"{op['path']}: 바꿀 칸이 없다")
            if parent[key] == op["value"]:
                raise ValueError(f"{op['path']}: 이미 같은 값이라 바뀌는 것이 없다")
            parent[key] = copy.deepcopy(op["value"])
        elif kind == "append":
            names = {x.get("name") for x in parent[key]}
            if op["value"].get("name") in names:
                raise ValueError(f"{op['path']}: 같은 이름 {op['value']['name']}가 이미 있다")
            parent[key].append(copy.deepcopy(op["value"]))
        elif kind == "insert_after":
            lst = parent[key]
            idx = next(i for i, x in enumerate(lst) if x.get("name") == op["after"])
            lst.insert(idx + 1, copy.deepcopy(op["value"]))
        elif kind == "remove":
            if key not in parent:
                raise KeyError(f"{op['path']}: 지울 칸이 없다")
            del parent[key]
        else:
            raise ValueError(f"모르는 명령: {kind}")
    return d


def targets_of(ops: list[dict]) -> list[str]:
    """패치가 건드리는 설계서 항목 (채점의 지목 대상, 같은 변형 안 충돌 검사)."""
    out = []
    for op in ops:
        first = op["path"].split(".")[0]
        head, item = _STEP.match(first).groups()
        if op["op"] in ("append", "insert_after"):
            t = f"{op['path']}:{op['value']['name']}"
        elif item is not None:
            t = f"{head}:{item}"
        elif head == "outcome":
            t = op["path"]
        else:
            t = head
        if t not in out:
            out.append(t)
    return out


# --- 공개 12개 사례 패치 (designs/error_catalog_public.csv의 how_to_inject) ----------
# 항목 이름은 기본 설계의 다른 항목과 같은 방식으로 붙인다 (결함을 암시하지 않게).

CR = {"test": ["creatinine"]}

PUBLIC_PATCHES: dict[str, list[dict]] = {
    "E01": [{"op": "append", "path": "features",
             "value": {"name": "n_dx", "source": "diagnoses", "agg": "count"}}],
    "E02": [{"op": "set", "path": "features[name=cr_last].time_column", "value": "collect_time"}],
    "E03": [{"op": "append", "path": "features",
             "value": {"name": "cr_max_adm", "source": "labs", "filter": CR, "time_column": "report_time",
                       "window": {"start": "admit", "end": "discharge"}, "agg": "max"}}],
    "E06": [{"op": "set", "path": "split.key", "value": "admission_id"}],
    # 기본 설계의 독립 단위가 이미 가족이므로 분할 키만 환자로 둔다.
    "E07": [{"op": "set", "path": "split.key", "value": "patient_id"}],
    "E08": [{"op": "set", "path": "preprocessing[name=median_impute].fit_scope", "value": "all"}],
    "E10": [{"op": "append", "path": "preprocessing",
             "value": {"name": "select_k", "kind": "select", "k": 5, "fit_scope": "all"}}],
    "E12": [{"op": "append", "path": "features",
             "value": {"name": "ward_type", "source": "admissions", "column": "unit",
                       "scope": "next_admission", "agg": "value"}}],
    "E13": [{"op": "append", "path": "features",
             "value": {"name": "renal_ordered", "source": "orders",
                       "filter": {"order_type": ["dialysis_order", "nephrology_consult"]},
                       "time_column": "order_time", "window": {"start": "admit", "end": "tp"}, "agg": "any"}}],
    "E14": [{"op": "append", "path": "cohort.inclusion",
             "value": {"name": "los_7d", "source": "admissions", "column": "length_of_stay_h",
                       "agg": "value", "op": ">=", "value": 168}}],
    "E15": [{"op": "append", "path": "cohort.inclusion",
             "value": {"name": "cr_monitored", "source": "labs", "filter": CR, "time_column": "report_time",
                       "window": {"start": "admit", "end": "discharge"}, "agg": "count", "op": ">=",
                       "value": 3}}],
    "E17": [{"op": "remove", "path": "outcome.ascertainment"}],
}


# --- 수상해 보이지만 정당한 변경 (3단계 시험 tests/design_mutations.TRICKY_DYNAMIC) -----------
# 고정 시점 설계에는 그 설계에서도 원리상 정당한 것만 쓴다 (tₚ = 퇴원 시각).

LEGIT_CHANGES: dict[str, dict] = {
    "prior_admission_dx": {"types": ["dynamic", "fixed"], "ops": [
        {"op": "append", "path": "features",
         "value": {"name": "prior_dx_renal_cardiac", "source": "diagnoses", "scope": "prior_admissions",
                   "filter": {"icd_code": ["N17", "N18", "I50"]}, "agg": "any"}}]},
    "report_time_lab_near_tp": {"types": ["dynamic"], "ops": [
        {"op": "append", "path": "features",
         "value": {"name": "k_last_6h", "source": "labs", "filter": {"test": ["potassium"]},
                   "time_column": "report_time", "window": {"start": "tp-6h", "end": "tp"}, "agg": "last"}}]},
    "report_time_lab_prior_admission": {"types": ["dynamic", "fixed"], "ops": [
        {"op": "append", "path": "features",
         "value": {"name": "prior_cr_last", "source": "labs", "scope": "prior_admissions", "filter": CR,
                   "time_column": "report_time", "window": {"end": "tp"}, "agg": "last"}}]},
    "admissions_before_tp": {"types": ["dynamic"], "ops": [
        {"op": "append", "path": "features",
         "value": {"name": "n_admissions_to_tp", "source": "admissions", "scope": "patient_history",
                   "time_column": "admit_time", "window": {"end": "tp"}, "agg": "count"}}]},
}


# --- 사례 목록 ---------------------------------------------------------------

def public_cases(catalog_csv: Path = ROOT / "designs" / "error_catalog_public.csv") -> list[dict]:
    import csv
    with open(catalog_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [case_entry(r, PUBLIC_PATCHES[r["id"]], holdout=False) for r in rows]


def case_entry(row: dict, ops: list[dict], holdout: bool, adjustment: str | None = None) -> dict:
    return {"id": row["id"], "name": row["name"], "question": row["question"],
            "types": TYPE_KO[row["design_types"]], "expected_verdict": row["expected_verdict"],
            "holdout": holdout, "adjustment": adjustment, "ops": ops, "targets": targets_of(ops)}


# --- 배치 ------------------------------------------------------------------

def target_counts(cases: list[dict], total: int, rng: random.Random) -> dict[str, int]:
    """사례별 배치 횟수를 고르게(차이 ≤ 1). 공개·보류 비율을 유지해 추가 배치를 나눈다."""
    n = len(cases)
    base, extra = divmod(total, n)
    counts = {c["id"]: base for c in cases}
    pub = [c["id"] for c in cases if not c["holdout"]]
    hold = [c["id"] for c in cases if c["holdout"]]
    extra_hold = round(extra * len(hold) / n)
    for ids, k in ((hold, extra_hold), (pub, extra - extra_hold)):
        for cid in rng.sample(sorted(ids), k):
            counts[cid] += 1
    return counts


def _feasible_pair(a: dict, b: dict) -> bool:
    return (a["id"] != b["id"] and a["id"] not in SOLO_CASES and b["id"] not in SOLO_CASES
            and not set(a["targets"]) & set(b["targets"]))


def assign(cases: list[dict], rng: random.Random, total: int = 24, tries: int = 20000) -> dict[str, list[list[str]]]:
    """유형별 결함 변형 N_DEFECT_VARIANTS개에 사례를 배치한다. 반환: {유형: [[사례 id, ...], ...]}."""
    by_id = {c["id"]: c for c in cases}
    for _ in range(tries):
        counts = target_counts(cases, total, rng)
        # 각 배치를 유형에 나눈다 (둘다 사례는 두 번 배치되면 두 유형에 하나씩)
        slots: dict[str, list[str]] = {"dynamic": [], "fixed": []}
        for cid in sorted(counts):
            types = list(by_id[cid]["types"])
            rng.shuffle(types)
            for i in range(counts[cid]):
                slots[types[i % len(types)]].append(cid)
        out = {}
        for t, ids in slots.items():
            layout = _pack(ids, by_id, rng)
            if layout is None:
                break
            out[t] = layout
        else:
            return out
    raise RuntimeError("배치 제약을 만족하는 배치를 찾지 못했다")


def _pack(ids: list[str], by_id: dict, rng: random.Random) -> list[list[str]] | None:
    n_pairs = len(ids) - N_DEFECT_VARIANTS
    if not 0 <= n_pairs <= N_DEFECT_VARIANTS:
        return None
    ids = ids[:]
    rng.shuffle(ids)
    pairs, rest = ids[: 2 * n_pairs], ids[2 * n_pairs:]
    groups = [pairs[i: i + 2] for i in range(0, len(pairs), 2)] + [[x] for x in rest]
    for g in groups:
        if len(g) == 2 and not _feasible_pair(by_id[g[0]], by_id[g[1]]):
            return None
    rng.shuffle(groups)
    return groups


# --- 변형 만들기 -------------------------------------------------------------

def _code(rng: random.Random, used: set[str]) -> str:
    while True:
        c = "".join(rng.choice("0123456789ABCDEF") for _ in range(4))
        if c not in used:
            used.add(c)
            return c


def _blind(d: dict, code: str, rng: random.Random) -> dict:
    d = copy.deepcopy(d)
    d["design_id"] = f"design_{code}"
    rng.shuffle(d["features"])
    rng.shuffle(d["cohort"]["inclusion"])
    return d


def build(bases: dict[str, dict], cases: list[dict], seed: int) -> tuple[dict[str, dict], dict]:
    """→ (변형 {파일명: 설계서}, 정답표). 같은 입력·시드 → 같은 결과."""
    rng = random.Random(seed)
    by_id = {c["id"]: c for c in cases}
    layout = assign(cases, rng)
    used: set[str] = set()
    variants, key = {}, {}
    for t in ("dynamic", "fixed"):
        legit_name = rng.choice(sorted(n for n, v in LEGIT_CHANGES.items() if t in v["types"]))
        plans = [("defect", g) for g in layout[t]] + [("clean", []), ("legit", [legit_name])]
        rng.shuffle(plans)
        for kind, group in plans:
            code = _code(rng, used)
            fname = f"design_{code}.json"
            d = bases[t]
            entry = {"design_type": t, "defects": [], "legit_changes": []}
            if kind == "defect":
                for cid in group:
                    c = by_id[cid]
                    d = apply_ops(d, c["ops"])
                    entry["defects"].append({k: c[k] for k in ("id", "name", "question", "expected_verdict",
                                                               "holdout", "adjustment", "targets")})
            elif kind == "legit":
                ops = LEGIT_CHANGES[legit_name]["ops"]
                d = apply_ops(d, ops)
                entry["legit_changes"].append({"name": legit_name, "targets": targets_of(ops),
                                               "note": "결함 아님 (원리상 정당한 변경). 지적하면 오경보"})
            v = _blind(d, code, rng)
            variants[fname] = v
            entry["sha256"] = file_sha256(dumps(v))
            key[fname] = entry
    return variants, key


def dumps(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, indent=2) + "\n"


def file_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_bases() -> dict[str, dict]:
    return {t: json.loads((BASE_DIR / f).read_text(encoding="utf-8")) for t, f in BASE_FILES.items()}


def placement_totals(key: dict) -> dict[str, int]:
    """공개·보류 배치 합계 (어느 파일인지는 담지 않는다)."""
    c = Counter("holdout" if d["holdout"] else "public" for v in key.values() for d in v["defects"])
    return {"public": c["public"], "holdout": c["holdout"]}


def seal_key(payload: dict, password: str) -> bytes:
    blob = encrypt_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), password)
    assert json.loads(decrypt_bytes(blob, password)) == payload  # 봉인 직후 메모리에서 확인
    return blob


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    if ANSWER_KEY.exists():
        raise SystemExit(f"{ANSWER_KEY}가 이미 있다. 주입은 한 번만 한다.")
    if VARIANT_DIR.exists() and any(VARIANT_DIR.iterdir()):
        raise SystemExit(f"{VARIANT_DIR}가 비어 있지 않다.")

    # 보류 사례: [{"row": 목록 행, "ops": [...], "adjustment": 조정 내용 또는 null}, ...]
    holdout_in = json.loads(sys.stdin.read())
    password = get_password()
    cases = public_cases() + [case_entry(h["row"], h["ops"], True, h.get("adjustment")) for h in holdout_in]
    ids = [c["id"] for c in cases]
    if len(ids) != 18 or len(set(ids)) != 18:
        raise SystemExit(f"사례가 18개가 아니다: {sorted(ids)}")

    seed = secrets.randbits(32)
    bases = load_bases()
    variants, key = build(bases, cases, seed)

    payload = {
        "created": dt.date.today().isoformat(),
        "seed": seed,
        "nonce": secrets.token_hex(16),
        "base_sha256": {t: file_sha256((BASE_DIR / f).read_text(encoding="utf-8")) for t, f in BASE_FILES.items()},
        "variants": key,
        "placement_counts": dict(sorted(Counter(d["id"] for v in key.values() for d in v["defects"]).items())),
        "holdout_cases": [{**h["row"], "ops": h["ops"], "adjustment": h.get("adjustment")} for h in holdout_in],
        "holdout_adjusted": {h["row"]["id"]: h["adjustment"] for h in holdout_in if h.get("adjustment")},
    }
    blob = seal_key(payload, password)

    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, v in variants.items():
        (VARIANT_DIR / fname).write_text(dumps(v), encoding="utf-8")
    ANSWER_KEY.write_bytes(blob)
    print(f"변형 {len(variants)}개 → {VARIANT_DIR.relative_to(ROOT)}")
    print(f"정답표 → {ANSWER_KEY.relative_to(ROOT)} (암호화)")
    print(f"배치 합계: {placement_totals(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
