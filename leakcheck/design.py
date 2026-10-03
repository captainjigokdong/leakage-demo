"""설계서 읽기와 형식 검사 (designs/schema.json)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema

from leakcheck import rules, timeline

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "designs" / "schema.json"
PROVENANCE_OK = "leakcheck.features"


class DesignError(ValueError):
    pass


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(design: dict) -> list[str]:
    v = jsonschema.Draft202012Validator(load_schema())
    errs = [f"{'/'.join(map(str, e.absolute_path)) or '(최상위)'}: {e.message}"
            for e in sorted(v.iter_errors(design), key=lambda e: list(map(str, e.absolute_path)))]
    for where, spec in _windows(design):
        for k in ("start", "end"):
            if k in spec:
                try:
                    timeline.parse(spec[k])
                except ValueError as ex:
                    errs.append(f"{where}.{k}: {ex}")
    if design.get("design_type") == "fixed" and len(design.get("tp", {}).get("offsets_h", [])) != 1:
        errs.append("tp.offsets_h: 고정 시점 설계는 예측 시점이 하나여야 한다")
    return errs


def _windows(d: dict):
    o = d.get("outcome", {})
    for k in ("window", "reference_window"):
        if k in o:
            yield f"outcome.{k}", o[k]
    for f in d.get("features", []):
        if "window" in f:
            yield f"features[{f.get('name')}].window", f["window"]
    c = d.get("cohort", {})
    for part in ("inclusion", "exclusion"):
        for f in c.get(part, []):
            if "window" in f:
                yield f"cohort.{part}[{f.get('name')}].window", f["window"]


def default_scope(source: str | None) -> str:
    return "patient" if source == "patients" else "index_admission"


def normalize(design: dict) -> dict:
    """형식 검사 후 기본값을 채운 복사본."""
    errs = validate(design)
    if errs:
        raise DesignError("설계서 형식 오류:\n" + "\n".join(f"- {e}" for e in errs))
    d = copy.deepcopy(design)
    d.setdefault("split_unit", rules.DEFAULT_SPLIT_UNIT)
    d.setdefault("cohort", {})
    d["cohort"].setdefault("index_time", "tp")
    d["cohort"].setdefault("inclusion", [])
    d["cohort"].setdefault("exclusion", [])
    d["cohort"].setdefault("subgroups", [])
    d.setdefault("preprocessing", [])
    d.setdefault("model", {})
    o = d["outcome"]
    o.setdefault("scope", default_scope(o["source"]))
    o.setdefault("filter", {})
    o.setdefault("proxies", [])
    o.setdefault("ascertainment", {})
    for spec in d["features"] + d["cohort"]["inclusion"] + d["cohort"]["exclusion"]:
        if "source" in spec:
            spec.setdefault("scope", default_scope(spec["source"]))
            spec.setdefault("filter", {})
            spec.setdefault("agg", "value" if spec["scope"] in ("patient", "index_admission")
                            and spec["source"] in ("patients", "admissions") else "last")
            spec.setdefault("made_by", PROVENANCE_OK)
    return d


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def design_hash(design: dict) -> str:
    raw = json.dumps(design, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def provenance_known(spec: dict) -> bool:
    return "source" in spec and spec.get("made_by", PROVENANCE_OK) == PROVENANCE_OK


def frames(design: dict) -> list[timeline.Frame]:
    tp = design["tp"]
    return [timeline.Frame(tp["anchor"], o) for o in tp["offsets_h"]]
