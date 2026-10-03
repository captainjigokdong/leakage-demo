"""tₚ 기준 기호 시각.

설계서의 시각은 "기준점 + 간격" 문자열로 적는다: `tp`, `tp-48h`, `admit+24h`, `discharge+30d`,
`-inf`, `inf`. 기준점은 예측 행이 속한 입원(index admission)의 입원·퇴원 시각과 tₚ다.

설계서만으로 시각을 비교하기 위해, 랜드마크 하나마다 각 기준점이 tₚ보다 몇 시간 앞뒤인지의
범위 [lo, hi]를 정한다.

- tₚ = 입원 + o (동적): admit = -o, discharge ∈ (0, ∞)  ← 입원 중인 tₚ만 예측 행이 된다
- tₚ = 퇴원 + o (고정): discharge = -o, admit ∈ (-∞, -o)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

INF = float("inf")
ANCHORS = ("tp", "admit", "discharge")

_RE = re.compile(r"^\s*(tp|admit|discharge)\s*(?:([+-])\s*(\d+(?:\.\d+)?)\s*([hd]))?\s*$")


@dataclass(frozen=True)
class TimeExpr:
    anchor: str      # tp | admit | discharge | -inf | inf
    offset_h: float = 0.0

    def __str__(self) -> str:
        if self.anchor in ("-inf", "inf") or self.offset_h == 0:
            return self.anchor
        sign = "+" if self.offset_h > 0 else "-"
        h = abs(self.offset_h)
        txt = f"{h / 24:g}d" if h % 24 == 0 else f"{h:g}h"
        return f"{self.anchor}{sign}{txt}"


def parse(s: str) -> TimeExpr:
    s = str(s).strip()
    if s in ("-inf", "inf"):
        return TimeExpr(s)
    m = _RE.match(s)
    if not m:
        raise ValueError(f"시각 표현을 읽을 수 없다: '{s}' (예: tp, tp-48h, admit+24h, discharge+30d, -inf)")
    anchor, sign, num, unit = m.groups()
    off = 0.0
    if num is not None:
        off = float(num) * (24 if unit == "d" else 1) * (-1 if sign == "-" else 1)
    return TimeExpr(anchor, off)


class Frame:
    """랜드마크 하나에서 기준점들의 tₚ 대비 범위(시간)."""

    def __init__(self, tp_anchor: str, tp_offset_h: float):
        if tp_anchor not in ("admit", "discharge"):
            raise ValueError(f"tₚ 기준점은 admit 또는 discharge여야 한다: {tp_anchor}")
        self.tp_anchor = tp_anchor
        self.o = float(tp_offset_h)

    def label(self) -> str:
        return str(TimeExpr(self.tp_anchor, self.o))

    def bounds(self, anchor: str) -> tuple[float, float]:
        if anchor == "tp":
            return 0.0, 0.0
        if anchor == "-inf":
            return -INF, -INF
        if anchor == "inf":
            return INF, INF
        if self.tp_anchor == "admit":
            if anchor == "admit":
                return -self.o, -self.o
            if anchor == "discharge":
                return 0.0, INF
        else:
            if anchor == "discharge":
                return -self.o, -self.o
            if anchor == "admit":
                return -INF, -self.o
        raise ValueError(f"알 수 없는 기준점: {anchor}")

    def lo(self, e: TimeExpr | str) -> float:
        e = parse(e) if isinstance(e, str) else e
        return self.bounds(e.anchor)[0] + e.offset_h

    def hi(self, e: TimeExpr | str) -> float:
        e = parse(e) if isinstance(e, str) else e
        return self.bounds(e.anchor)[1] + e.offset_h


def fmt_h(x: float) -> str:
    """tₚ 대비 시간을 읽기 좋게."""
    if x == INF:
        return "없음(tₚ 뒤일 수 있음)"
    if x == -INF:
        return "-∞"
    if x == 0:
        return "tₚ"
    return f"tₚ{'+' if x > 0 else ''}{x:g}h"


# --- 데이터 단계 ---

T_MIN = pd.Timestamp.min
T_MAX = pd.Timestamp.max


def resolve(e: TimeExpr | str, index_rows: pd.DataFrame) -> pd.Series:
    """예측 행마다 실제 시각. -inf/inf는 Timestamp.min/max."""
    e = parse(e) if isinstance(e, str) else e
    n = len(index_rows)
    if e.anchor == "-inf":
        return pd.Series([T_MIN] * n, index=index_rows.index, dtype="datetime64[ns]")
    if e.anchor == "inf":
        return pd.Series([T_MAX] * n, index=index_rows.index, dtype="datetime64[ns]")
    col = {"tp": "tp", "admit": "admit_time", "discharge": "discharge_time"}[e.anchor]
    return index_rows[col] + pd.Timedelta(hours=e.offset_h)
