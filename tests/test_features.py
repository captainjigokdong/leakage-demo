import pandas as pd
import pytest

from leakcheck import design as dz
from leakcheck.features import UNKNOWN, build_index, feature_matrix, make_feature
from leakcheck.tagging import tag_tables
from tests.design_mutations import clean


@pytest.fixture(scope="module")
def ctx(small_tables):
    d = dz.normalize(clean("동적"))
    tagged = tag_tables(small_tables)
    return d, tagged, build_index(d, tagged)


def _spec(d, name):
    return next(f for f in d["features"] if f["name"] == name)


def test_dynamic_index_rows_only_while_admitted(ctx):
    _, _, idx = ctx
    assert (idx["tp"] < idx["discharge_time"]).all()
    assert (idx["tp"] == idx["admit_time"] + pd.to_timedelta(idx["landmark_h"], unit="h")).all()
    assert idx["index_id"].is_unique


def test_feature_inherits_tags_from_source_rows(ctx):
    d, tagged, idx = ctx
    r = make_feature(tagged, _spec(d, "cr_max_48h"), idx)
    labs = tagged["labs"].set_index("_provenance")
    fr = r.frame.set_index("index_id")
    row = fr[fr["provenance"].map(len) >= 2].iloc[0]
    src = labs.loc[list(row["provenance"])]
    assert (src["test"] == "creatinine").all()
    assert row["value"] == src["value"].max()
    tp = idx.set_index("index_id").loc[row.name, "tp"]
    # 확인 가능 시각 = max(재료 행의 보고 시각, 창 끝 tₚ)
    assert row["available_time"] == max(src["report_time"].max(), tp)
    assert "max(labs.value" in r.method and "report_time" in r.method


def test_empty_window_is_known_at_window_end(ctx):
    d, tagged, idx = ctx
    spec = dict(_spec(d, "n_cr"), window={"start": "tp-6h", "end": "tp"})
    r = make_feature(tagged, spec, idx).frame.set_index("index_id")
    empty = r[r["value"] == 0]
    assert len(empty) > 0
    assert (empty["available_time"] == idx.set_index("index_id").loc[empty.index, "tp"]).all()
    assert (empty["provenance"].map(len) == 0).all()


def test_report_time_window_never_leaks(ctx):
    d, tagged, idx = ctx
    for f in d["features"]:
        assert not make_feature(tagged, f, idx).leaks(idx).any(), f["name"]


def test_collect_time_window_leaks_in_data(ctx):
    d, tagged, idx = ctx
    spec = dict(_spec(d, "cr_last"), time_column="collect_time")
    assert make_feature(tagged, spec, idx).leaks(idx).sum() > 0


def test_registry_marks_unregistered_columns_unknown(ctx):
    d, tagged, idx = ctx
    X, reg = feature_matrix([make_feature(tagged, _spec(d, "age"), idx)], idx)
    X["hand_made"] = 1.0
    assert reg.unknown_columns(X.columns.drop("index_id")) == ["hand_made"]
    assert reg.provenance_of("hand_made") == UNKNOWN
    assert reg.provenance_of("age").startswith("value(patients.age")
