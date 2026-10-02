import gzip
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth.generate import DEFAULT_OUT, TABLES, generate, load, save, to_csv_bytes
from synth.stats import check_criteria, kdigo_first, kdigo_table, readmission_table, summarize

SMALL = 300
SEED = 7


@pytest.fixture(scope="module")
def ehr():
    return generate(seed=SEED, n_patients=SMALL)


@pytest.fixture(scope="module")
def stats(ehr):
    return summarize(ehr.tables)


# --- 재현성 ---

def test_same_seed_same_data(ehr):
    again = generate(seed=SEED, n_patients=SMALL)
    for name in TABLES:
        assert to_csv_bytes(ehr.tables[name]) == to_csv_bytes(again.tables[name]), name


def test_different_seed_different_data(ehr):
    other = generate(seed=SEED + 1, n_patients=SMALL)
    assert to_csv_bytes(ehr.tables["labs"]) != to_csv_bytes(other.tables["labs"])


def test_n_patients_is_configurable():
    assert len(generate(seed=1, n_patients=50).tables["patients"]) == 50
    with pytest.raises(ValueError):
        generate(seed=1, n_patients=0)


def test_save_load_roundtrip(ehr, tmp_path):
    manifest = save(ehr, tmp_path)
    loaded = load(tmp_path)
    for name in TABLES:
        assert manifest["tables"][name]["rows"] == len(ehr.tables[name])
        assert to_csv_bytes(loaded[name]) == to_csv_bytes(ehr.tables[name]), name
    # 압축 파일도 바이트 단위로 같아야 한다 (gzip mtime 고정)
    first = (tmp_path / "labs.csv.gz").read_bytes()
    save(ehr, tmp_path)
    assert (tmp_path / "labs.csv.gz").read_bytes() == first


# --- 구조 ---

def test_schema(ehr):
    t = ehr.tables
    assert list(t["patients"].columns) == ["patient_id", "family_id", "age", "sex"]
    assert list(t["admissions"].columns) == [
        "admission_id", "patient_id", "admit_time", "discharge_time", "discharge_status", "unit"]
    assert list(t["labs"].columns) == [
        "lab_id", "admission_id", "test", "value", "unit", "collect_time", "report_time"]
    assert list(t["orders"].columns) == ["order_id", "admission_id", "order_type", "order_time"]
    assert set(t["admissions"]["unit"]) == {"ICU", "ward"}
    assert set(t["labs"]["test"]) == {"creatinine", "bun", "potassium", "hemoglobin"}


def test_keys_unique_and_foreign_keys_exist(ehr):
    t = ehr.tables
    for name, key in (("patients", "patient_id"), ("admissions", "admission_id"),
                      ("labs", "lab_id"), ("orders", "order_id")):
        assert t[name][key].is_unique, name
    assert t["admissions"]["patient_id"].isin(t["patients"]["patient_id"]).all()
    for name in ("labs", "diagnoses", "procedures", "orders"):
        assert t[name]["admission_id"].isin(t["admissions"]["admission_id"]).all(), name


def test_diagnoses_have_no_time(ehr):
    cols = ehr.tables["diagnoses"].columns
    assert not any("time" in c or "date" in c for c in cols)


def test_procedures_have_date_only(ehr):
    p = ehr.tables["procedures"]
    assert not any("time" in c for c in p.columns)
    assert p["chart_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()


def test_admission_times_ordered(ehr):
    a = ehr.tables["admissions"].sort_values(["patient_id", "admit_time"])
    assert (a["discharge_time"] > a["admit_time"]).all()
    prev_dis = a.groupby("patient_id")["discharge_time"].shift(1)
    assert (a["admit_time"][prev_dis.notna()] > prev_dis.dropna()).all()  # 입원이 겹치지 않음
    # 사망 퇴원 뒤에는 입원이 없다
    last = a.groupby("patient_id").tail(1)
    died = a[a["discharge_status"] == "died"]
    assert died["admission_id"].isin(last["admission_id"]).all()


# --- 시각 성질 (Q1) ---

def test_report_after_collect(ehr):
    labs = ehr.tables["labs"]
    delay = labs["report_time"] - labs["collect_time"]
    assert (delay >= pd.Timedelta(minutes=30)).all()
    assert (delay <= pd.Timedelta(hours=6)).all()


def test_labs_collected_during_stay(ehr):
    m = ehr.tables["labs"].merge(ehr.tables["admissions"], on="admission_id")
    assert (m["collect_time"] >= m["admit_time"]).all()
    assert (m["collect_time"] <= m["discharge_time"]).all()


def test_orders_during_stay(ehr):
    m = ehr.tables["orders"].merge(ehr.tables["admissions"], on="admission_id")
    assert ((m["order_time"] >= m["admit_time"]) & (m["order_time"] <= m["discharge_time"])).all()


def test_post_discharge_reported_labs_exist(stats):
    assert stats["post_discharge_lab_n"] > 0


# --- 측정 강도 (Q7) ---

def test_icu_measures_more_often(stats):
    assert stats["cr_per_day_icu"] > 1.5 * stats["cr_per_day_ward"]


def test_ward_misses_more_latent_aki():
    """측정이 드문 병동에서 잠재 AKI가 KDIGO로 덜 잡힌다. 표본이 커야 안정적이라 1,000명으로 본다."""
    e = generate(seed=SEED, n_patients=1000)
    m = (e.tables["admissions"].merge(kdigo_table(e.tables["labs"]), on="admission_id")
         .merge(e.latent, on="admission_id"))
    m = m[m["aki_latent"]]
    detect = m.groupby("unit")["aki"].mean()
    assert detect["ICU"] > detect["ward"]


# --- KDIGO 판정 ---

def test_kdigo_absolute_rise_within_48h():
    assert kdigo_first(np.array([0, 24.0]), np.array([1.0, 1.3])) == 1
    assert kdigo_first(np.array([0, 24.0]), np.array([1.0, 1.29])) is None
    assert kdigo_first(np.array([0, 50.0]), np.array([1.0, 1.3])) is None  # 48시간 밖


def test_kdigo_relative_rise_within_7_days():
    assert kdigo_first(np.array([0, 100.0]), np.array([0.6, 0.9])) == 1  # 1.5배
    assert kdigo_first(np.array([0, 24 * 8.0]), np.array([0.6, 0.9])) is None  # 7일 밖


def test_kdigo_single_measurement_is_not_aki():
    assert kdigo_first(np.array([0.0]), np.array([5.0])) is None


def test_readmission_definition():
    t = pd.Timestamp("2150-01-01")
    adm = pd.DataFrame({
        "admission_id": ["A1", "A2", "A3", "B1", "B2"],
        "patient_id": ["P1", "P1", "P1", "P2", "P2"],
        "admit_time": [t, t + pd.Timedelta(days=20), t + pd.Timedelta(days=100), t, t + pd.Timedelta(days=50)],
        "discharge_time": [t + pd.Timedelta(days=3), t + pd.Timedelta(days=25), t + pd.Timedelta(days=105),
                           t + pd.Timedelta(days=2), t + pd.Timedelta(days=55)],
        "discharge_status": ["home", "home", "died", "home", "home"],
        "unit": ["ward"] * 5,
    })
    r = readmission_table(adm).set_index("admission_id")["readmit30"]
    assert r.to_dict() == {"A1": True, "A2": False, "B1": False, "B2": False}  # A3 사망은 분모에서 제외


# --- 함정이 들어 있는지 (작은 데이터라 기준을 넓게) ---

def test_pitfalls_present_in_small_data(ehr, stats):
    assert stats["multi_admit_frac"] >= 0.15
    assert stats["post_discharge_lab_frac"] >= 0.01
    assert stats["family_pairs"] >= 15
    assert stats["icu_patients"] > 0 and stats["ward_patients"] > 0
    assert stats["aki_patients_with_renal_order"] >= 1
    assert 0.05 <= stats["aki_rate"] <= 0.25
    assert 0.05 <= stats["readmit30_rate"] <= 0.25


def test_family_pairs_counted_correctly(ehr, stats):
    sizes = ehr.tables["patients"]["family_id"].dropna().value_counts()
    assert set(sizes) <= {2, 3}
    assert stats["family_pairs"] == sum(comb(int(s), 2) for s in sizes)


def test_aki_admissions_get_renal_proxies(ehr):
    """대리 변수(Q4)가 결과와 실제로 얽혀 있다: 신장내과 협진은 AKI 입원에서 더 흔하다."""
    t = ehr.tables
    kd = kdigo_table(t["labs"]).set_index("admission_id")["aki"]
    consult = t["orders"].loc[t["orders"]["order_type"] == "nephrology_consult", "admission_id"]
    has = kd.index.isin(consult)
    assert has[kd.to_numpy()].mean() > 3 * has[~kd.to_numpy()].mean()


# --- 저장된 기본 데이터 (5,000명) ---

@pytest.mark.skipif(not (DEFAULT_OUT / "MANIFEST.json").exists(), reason="기본 데이터가 아직 생성되지 않음")
def test_committed_data_matches_manifest_and_criteria():
    manifest = json.loads((DEFAULT_OUT / "MANIFEST.json").read_text())
    for name, info in manifest["tables"].items():
        raw = gzip.decompress((DEFAULT_OUT / info["file"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == info["sha256_csv"], name
    checks = check_criteria(summarize(load(DEFAULT_OUT)))
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
