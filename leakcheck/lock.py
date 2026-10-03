"""결정 카드와 결정 잠금. docs/research_plan.md 5절.

결과를 바꾸는 설계 결정(결과 정의, 포함·제외, 분할 단위 등)을 카드로 사람에게 보여 준다.
카드에는 기술 통계(표본 수, 결과율)만 있고 모델 성능은 없다. 사람이 승인하면 잠기고,
잠금은 설계서 해시에 묶인다. 성능 계산은 잠금을 거친 뒤에만 된다 (CLAUDE.md 절대 규칙 6).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from leakcheck import design as dz
from leakcheck import outcomes
from leakcheck.checks import prepare_data

# 카드에 들어가면 안 되는 성능 지표 이름 (부분 일치, 소문자)
FORBIDDEN_STATS = ("auc", "auroc", "auprc", "accuracy", "precision", "recall", "sensitivity",
                   "specificity", "f1", "brier", "calibration", "c_index", "c-index", "loss", "score")


class LockError(RuntimeError):
    """잠금 전에 성능을 계산하려 했거나, 잠근 뒤 설계서가 바뀌었다."""


class CardError(ValueError):
    """결정 카드에 성능 지표가 들어갔다."""


@dataclass
class DecisionCard:
    design_id: str
    design_hash: str
    decisions: dict
    stats: dict

    def __post_init__(self):
        bad = [k for k in _all_keys(self.stats) if any(w in k.lower() for w in FORBIDDEN_STATS)]
        if bad:
            raise CardError(f"결정 카드에는 기술 통계만 넣는다. 성능 지표로 보이는 항목: {bad}")

    def to_text(self) -> str:
        s = self.stats
        lines = [f"결정 카드 — {self.design_id} (설계서 해시 {self.design_hash[:12]})", "", "[결정]"]
        lines += [f"- {k}: {v}" for k, v in self.decisions.items()]
        lines += ["", "[기술 통계] (모델 성능 없음)",
                  f"- 예측 행 {s['n_index_rows']:,} · 입원 {s['n_admissions']:,} · 환자 {s['n_patients']:,}",
                  f"- 결과율 {s['outcome_rate']:.1%} ({s['n_outcome']:,}건)"]
        for k, v in s["excluded_by_criterion"].items():
            lines.append(f"- 기준 '{k}'로 빠진 예측 행 {v:,}")
        for col, by in s["by_subgroup"].items():
            for lev, x in by.items():
                lines.append(f"- {col}={lev}: 예측 행 {x['n']:,}, 결과율 {x['outcome_rate']:.1%}")
        return "\n".join(lines)


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _all_keys(v)


def decision_card(design: dict, tables: dict) -> DecisionCard:
    d = dz.normalize(design)
    ctx = prepare_data(d, tables)
    y = outcomes.label(d, ctx.tagged, ctx.cohort)
    rows = ctx.cohort.assign(y=y.to_numpy())

    excluded = {}
    from leakcheck.checks import _criterion_mask
    for part, sign in (("inclusion", False), ("exclusion", True)):
        for spec in d["cohort"][part]:
            if dz.provenance_known(spec):
                m = _criterion_mask(ctx.tagged, spec, ctx.base)
                excluded[f"{part}:{spec['name']}"] = int((m if sign else ~m).sum())

    by = {}
    for col in dict.fromkeys(["unit"] + d["cohort"]["subgroups"]):
        if col in rows:
            by[col] = {str(k): {"n": int(len(g)), "outcome_rate": float(g["y"].mean())}
                       for k, g in rows.groupby(col)}
    stats = {
        "n_index_rows": int(len(rows)),
        "n_admissions": int(rows["admission_id"].nunique()),
        "n_patients": int(rows["patient_id"].nunique()),
        "n_outcome": int(rows["y"].sum()),
        "outcome_rate": float(rows["y"].mean()) if len(rows) else float("nan"),
        "excluded_by_criterion": excluded,
        "by_subgroup": by,
    }
    o = d["outcome"]
    decisions = {
        "예측 시점": f"{d['tp']['anchor']} + {d['tp']['offsets_h']}h",
        "결과": f"{o['name']} ({o['definition']}), 결과 창 ({o['window'].get('start', '-inf')}, "
                f"{o['window'].get('end', 'inf')}]",
        "포함 기준": [c["name"] for c in d["cohort"]["inclusion"]],
        "제외 기준": [c["name"] for c in d["cohort"]["exclusion"]],
        "독립 단위 / 분할 키": f"{d['split_unit']} / {d['split']['key']}",
    }
    return DecisionCard(d["design_id"], dz.design_hash(design), decisions, stats)


@dataclass
class DecisionLock:
    card: DecisionCard
    locked: bool = False
    approved_by: str | None = None
    approved_at: str | None = None
    notes: list[str] = field(default_factory=list)

    def approve(self, approver: str) -> None:
        if not approver:
            raise LockError("승인자 이름이 필요하다")
        self.locked = True
        self.approved_by = approver
        self.approved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def check(self, design: dict) -> None:
        if not self.locked:
            raise LockError("결정이 잠기지 않았다. 결정 카드를 사람이 승인한 뒤에만 성능을 계산한다.")
        if dz.design_hash(design) != self.card.design_hash:
            raise LockError("잠근 뒤 설계서가 바뀌었다. 새 결정 카드를 만들어 다시 승인받아야 한다.")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DecisionLock":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["card"] = DecisionCard(**raw["card"])
        return cls(**raw)


def requires_lock(fn):
    """첫 두 인자로 (lock, design)을 받는 성능 함수를 잠금 확인으로 감싼다."""
    def wrapper(lock: DecisionLock | None, design: dict, *args, **kwargs):
        if lock is None:
            raise LockError("결정 잠금 없이 성능을 계산할 수 없다.")
        lock.check(design)
        return fn(lock, design, *args, **kwargs)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@requires_lock
def auroc(lock: DecisionLock, design: dict, y_true, y_score) -> float:
    """잠금을 거친 뒤에만 계산되는 AUROC."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))
