import copy

import pytest

from leakcheck.lock import CardError, DecisionCard, DecisionLock, LockError, auroc, decision_card
from tests.design_mutations import clean


@pytest.fixture(scope="module")
def card(small_tables):
    return decision_card(clean("고정"), small_tables)


def test_card_has_descriptive_stats_only(card):
    s = card.stats
    assert s["n_index_rows"] > 0 and 0 < s["outcome_rate"] < 1
    assert set(s["by_subgroup"]["unit"]) == {"ICU", "ward"}
    assert "결과율" in card.to_text() and "모델 성능 없음" in card.to_text()


def test_card_rejects_performance_numbers(card):
    with pytest.raises(CardError):
        DecisionCard(card.design_id, card.design_hash, card.decisions, {**card.stats, "auroc": 0.8})


def test_performance_before_lock_raises(card):
    d = clean("고정")
    with pytest.raises(LockError):
        auroc(None, d, [0, 1], [0.2, 0.9])
    with pytest.raises(LockError):
        auroc(DecisionLock(card), d, [0, 1], [0.2, 0.9])


def test_performance_after_lock_and_design_change(card, tmp_path):
    d = clean("고정")
    lock = DecisionLock(card)
    lock.approve("사용자")
    assert auroc(lock, d, [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    changed = copy.deepcopy(d)
    changed["outcome"]["window"]["end"] = "tp+60d"
    with pytest.raises(LockError, match="설계서가 바뀌었다"):
        auroc(lock, changed, [0, 1], [0.2, 0.9])
    lock.save(tmp_path / "lock.json")
    again = DecisionLock.load(tmp_path / "lock.json")
    assert again.locked and again.approved_by == "사용자"
    assert auroc(again, d, [0, 1], [0.2, 0.9]) == 1.0


def test_dynamic_card(small_tables):
    c = decision_card(clean("동적"), small_tables)
    assert c.stats["excluded_by_criterion"]["exclusion:aki_known_by_tp"] > 0
    assert 0 < c.stats["outcome_rate"] < 0.5
