from collections import Counter

import pytest

from tools.draw_holdout import (
    ALLOCATION,
    COLUMNS,
    PUBLIC_CSV,
    catalog_hash,
    read_catalog,
    stratified_draw,
    to_csv,
    validate,
)

EXPECTED_PUBLIC = {q: n - 1 for q, n in ALLOCATION.items()}


def fake_catalog() -> list[dict]:
    rows, i = [], 1
    for q, n in ALLOCATION.items():
        for _ in range(n):
            rows.append(
                {
                    "id": f"X{i:02d}",
                    "name": f"가짜 사례 {i}",
                    "source": "test",
                    "question": q,
                    "design_types": "둘다",
                    "how_to_inject": "none",
                    "expected_verdict": "경고" if q == "Q7" else "차단",
                }
            )
            i += 1
    return rows


def test_draw_is_stratified_and_reproducible():
    rows = fake_catalog()
    public, holdout = stratified_draw(rows, seed=123)
    assert len(public) == 12 and len(holdout) == 6
    assert Counter(r["question"] for r in holdout) == Counter(ALLOCATION.keys())
    assert dict(Counter(r["question"] for r in public)) == EXPECTED_PUBLIC
    assert stratified_draw(rows, seed=123) == (public, holdout)


def test_hash_recomputable_from_parts():
    rows = fake_catalog()
    public, holdout = stratified_draw(rows, seed=7)
    assert catalog_hash(public + holdout) == catalog_hash(rows)


def test_validate_rejects_wrong_allocation():
    with pytest.raises(ValueError):
        validate(fake_catalog()[1:])


def test_csv_roundtrip(tmp_path):
    rows = fake_catalog()
    p = tmp_path / "c.csv"
    p.write_text(to_csv(rows), encoding="utf-8")
    assert read_catalog(p) == rows


@pytest.mark.skipif(not PUBLIC_CSV.exists(), reason="추첨 전")
def test_public_catalog_matches_plan():
    import csv

    with open(PUBLIC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert len(rows) == 12
    assert dict(Counter(r["question"] for r in rows)) == EXPECTED_PUBLIC
