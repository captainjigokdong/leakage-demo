"""4단계 결함 주입·맹검 변형 시험.

보류 사례 내용은 쓰지 않는다. 배치 시험은 가짜 보류 사례 6개로 한다.
결함 변형에는 점검기를 돌리지 않는다 (성공 기준 고정 전, 2026-10-03 사용자 결정).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from designs import inject as inj
from leakcheck.checks import run_checks
from leakcheck.design import normalize, validate
from synth.generate import TABLES, load
from synth.outpatient import FILE as OUTPATIENT_FILE, generate_visits
from tests.design_mutations import clean, inject as old_inject
from tools.seal import SealError, decrypt_bytes

ROOT = Path(__file__).resolve().parent.parent
KO = {"dynamic": "동적", "fixed": "고정"}
CASE_ID = re.compile(r"\bE\d{2}\b")


def fake_holdout() -> list[dict]:
    """보류 6개 자리를 채우는 가짜 사례 (질문별 1개, 실제 보류 사례와 무관)."""
    specs = [
        ("X1", "Q1", "동적", [{"op": "append", "path": "features",
                              "value": {"name": "f1", "source": "labs", "agg": "last"}}]),
        ("X2", "Q2", "동적", [{"op": "set", "path": "split.method", "value": "random_rows"}]),
        ("X3", "Q3", "고정", [{"op": "set", "path": "preprocessing[name=standardize].fit_scope", "value": "all"}]),
        ("X4", "Q4", "동적", [{"op": "append", "path": "features",
                              "value": {"name": "f4", "source": "orders", "agg": "count"}}]),
        ("X5", "Q5", "고정", [{"op": "append", "path": "cohort.exclusion",
                              "value": {"name": "c5", "source": "patients", "column": "age",
                                        "agg": "value", "op": ">", "value": 90}}]),
        ("X6", "Q7", "고정", [{"op": "remove", "path": "outcome.ascertainment"}]),
    ]
    return [inj.case_entry({"id": i, "name": i, "question": q, "design_types": t, "expected_verdict": "차단"},
                           ops, holdout=True) for i, q, t, ops in specs]


@pytest.fixture(scope="module")
def cases():
    return inj.public_cases() + fake_holdout()


@pytest.fixture(scope="module")
def built(cases):
    return inj.build(inj.load_bases(), cases, seed=12345)


# --- 기본 설계 ---

@pytest.mark.parametrize("t", ["dynamic", "fixed"])
def test_base_is_family_split(t):
    b = inj.load_bases()[t]
    assert (b["split_unit"], b["split"]["key"], b["model"]["tuning"]["cv_key"]) == ("family", "family_id", "family_id")


@pytest.mark.parametrize("t", ["dynamic", "fixed"])
def test_base_passes_design_and_data(t, small_tables):
    b = inj.load_bases()[t]
    assert run_checks(b).problems() == []
    assert run_checks(b, small_tables).problems() == []


def test_fixed_base_records_ascertainment_by_discharge_status():
    assert inj.load_bases()["fixed"]["outcome"]["ascertainment"]["by_stratum"] == ["discharge_status"]


# --- 외래 방문 테이블 (별도 파일) ---

def test_load_reads_only_six_tables():
    assert (ROOT / "data" / "synth" / OUTPATIENT_FILE).exists()
    assert sorted(load()) == sorted(TABLES) and len(TABLES) == 6


def test_outpatient_is_deterministic_and_respects_readmission(small_tables):
    p, a = small_tables["patients"], small_tables["admissions"]
    v1, v2 = generate_visits(p, a, seed=1), generate_visits(p, a, seed=1)
    assert v1.equals(v2) and len(v1) > 0
    assert set(v1["patient_id"]) <= set(p["patient_id"])


# --- 패치 ---

@pytest.mark.parametrize("case_id", sorted(set(inj.PUBLIC_PATCHES) - {"E07"}))
def test_public_patch_matches_stage3_mutation(case_id):
    """공개 패치는 3단계 시험의 주입과 같다 (심은 항목 이름만 다를 수 있음)."""
    case = next(c for c in inj.public_cases() if c["id"] == case_id)

    def strip(d):
        d = normalize(d)
        d["design_id"] = ""
        for part in (d["features"], d["cohort"]["inclusion"], d["preprocessing"]):
            for x in part:
                x.pop("name", None)
        return d

    for t in case["types"]:
        assert strip(inj.apply_ops(clean(KO[t]), case["ops"])) == strip(old_inject(case_id, KO[t]))


def test_e07_on_family_base_keeps_family_unit_and_patient_key():
    b = inj.load_bases()["dynamic"]
    d = inj.apply_ops(b, inj.PUBLIC_PATCHES["E07"])
    assert (d["split_unit"], d["split"]["key"]) == ("family", "patient_id")


def test_apply_ops_refuses_no_op_and_missing():
    b = inj.load_bases()["dynamic"]
    with pytest.raises(ValueError):
        inj.apply_ops(b, [{"op": "set", "path": "split.key", "value": "family_id"}])
    with pytest.raises(KeyError):
        inj.apply_ops(b, [{"op": "set", "path": "features[name=nope].agg", "value": "max"}])


def test_targets():
    assert inj.targets_of(inj.PUBLIC_PATCHES["E02"]) == ["features:cr_last"]
    assert inj.targets_of(inj.PUBLIC_PATCHES["E08"]) == ["preprocessing:median_impute"]
    assert inj.targets_of(inj.PUBLIC_PATCHES["E06"]) == ["split"]
    assert inj.targets_of(inj.PUBLIC_PATCHES["E17"]) == ["outcome.ascertainment"]


def test_injected_names_follow_base_naming():
    """심은 항목 이름이 결함을 암시하는 낱말을 쓰지 않는다."""
    bad = re.compile(r"leak|next|future|after|post|outcome|target_leak|all_data|E\d\d", re.I)
    for ops in inj.PUBLIC_PATCHES.values():
        for op in ops:
            if "value" in op and isinstance(op["value"], dict):
                assert not bad.search(op["value"]["name"])


# --- 배치 ---

def test_layout_constraints(cases, built):
    _, key = built
    by_id = {c["id"]: c for c in cases}
    assert len(key) == 20
    for t in ("dynamic", "fixed"):
        vs = [v for v in key.values() if v["design_type"] == t]
        assert len(vs) == 10
        assert sum(1 for v in vs if v["defects"]) == 8
        clean_vs = [v for v in vs if not v["defects"]]
        assert sorted(len(v["legit_changes"]) for v in clean_vs) == [0, 1]
        for v in vs:
            assert len(v["defects"]) <= 2
            ids = [d["id"] for d in v["defects"]]
            for cid in ids:
                assert t in by_id[cid]["types"]
                if cid in inj.SOLO_CASES:
                    assert ids == [cid]
            tg = [x for d in v["defects"] for x in d["targets"]]
            assert len(tg) == len(set(tg))


def test_every_case_placed_evenly(cases, built):
    _, key = built
    counts = Counter(d["id"] for v in key.values() for d in v["defects"])
    assert set(counts) == {c["id"] for c in cases}
    assert max(counts.values()) - min(counts.values()) <= 1
    tot = inj.placement_totals(key)
    assert tot["public"] + tot["holdout"] == sum(counts.values())
    # 추가 배치는 공개·보류 비율대로 (12:6)
    assert tot["public"] == 2 * tot["holdout"]


def test_solo_case_is_alone():
    rows = [{"id": "E16", "name": "x", "question": "Q5", "design_types": "고정", "expected_verdict": "차단"}]
    solo = inj.case_entry(rows[0], [{"op": "append", "path": "cohort.inclusion", "value": {
        "name": "s", "source": "admissions", "column": "unit", "agg": "value", "op": "==", "value": "icu"}}], True)
    cases = inj.public_cases() + fake_holdout()[:5] + [solo]
    for seed in range(5):
        _, key = inj.build(inj.load_bases(), cases, seed)
        for v in key.values():
            ids = [d["id"] for d in v["defects"]]
            assert "E16" not in ids or ids == ["E16"]


def test_build_is_deterministic(cases):
    a = inj.build(inj.load_bases(), cases, seed=7)
    b = inj.build(inj.load_bases(), cases, seed=7)
    assert a == b


# --- 맹검 ---

def test_variants_are_blind_and_valid(built):
    variants, key = built
    for fname, v in variants.items():
        assert re.fullmatch(r"design_[0-9A-F]{4}\.json", fname)
        assert v["design_id"] == fname[:-5]
        text = inj.dumps(v)
        assert not CASE_ID.search(text)
        assert key[fname]["sha256"] == inj.file_sha256(text)
        assert validate(v) == []


def test_clean_variants_pass_checker(built, small_tables):
    """깨끗한 변형(정당한 변경 포함)만 점검기로 확인한다."""
    variants, key = built
    for fname, v in variants.items():
        if not key[fname]["defects"]:
            assert run_checks(v, small_tables).problems() == [], fname


def test_seal_key_roundtrip(built):
    _, key = built
    blob = inj.seal_key({"variants": key}, "pw")
    assert json.loads(decrypt_bytes(blob, "pw")) == {"variants": key}
    assert b"design_" not in blob
    with pytest.raises(SealError):
        decrypt_bytes(blob, "wrong")


# --- 커밋된 변형 ---

VARIANTS = sorted((ROOT / "designs" / "variants").glob("*.json"))


@pytest.mark.skipif(not VARIANTS, reason="변형이 아직 생성되지 않음")
def test_committed_variants():
    assert len(VARIANTS) == 20
    types = Counter()
    for p in VARIANTS:
        assert re.fullmatch(r"design_[0-9A-F]{4}\.json", p.name)
        text = p.read_text(encoding="utf-8")
        assert not CASE_ID.search(text)
        d = json.loads(text)
        assert d["design_id"] == p.stem
        assert validate(d) == []
        types[d["design_type"]] += 1
    assert types == {"dynamic": 10, "fixed": 10}
    assert (ROOT / "sealed" / "answer_key.enc").exists()
