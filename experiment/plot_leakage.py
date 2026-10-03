"""누수 효과 시연 그림: results/leakage_effect.json → results/leakage_effect.png

설계(수정 / 누수 하나씩 / 모두)마다 모델별 시험 AUROC. 점 = 시드 평균, 가로선 = 시드 5개의 범위(로지스틱·부스팅), TPOT는 시드 1개.
그림 글자는 영어(컨테이너에 한글 글꼴 없음). 같은 수치의 표는 results/report.md에 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "results" / "leakage_effect.json", ROOT / "results" / "leakage_effect.png"

# 범주 색 (dataviz 기본 팔레트 1~3번, 검증 통과. 3번은 대비 3:1 미만이라 모양·표로 보완)
MODELS = [("logistic", "Logistic regression", "#2a78d6", "o"),
          ("hgb", "HistGradientBoosting", "#eb6834", "s"),
          ("tpot", "TPOT (1 seed)", "#1baf7a", "D")]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TITLES = {"fixed": "Fixed-time: 30-day readmission", "dynamic": "Dynamic: AKI within 48 h"}


def main() -> int:
    res = json.loads(SRC.read_text(encoding="utf-8"))
    summ = res["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, facecolor=SURFACE)
    for ax, t in zip(axes, ("fixed", "dynamic")):
        ax.set_facecolor(SURFACE)
        names = ["fixed"] + list(res["leaks"][t]) + ["all"]
        labels = ["Corrected (base)"] + [f"{n} only" for n in res["leaks"][t]] + ["All leaks"]
        for i, n in enumerate(names):
            y = len(names) - 1 - i
            for j, (m, _, color, marker) in enumerate(MODELS):
                s = summ[t][n].get(m)
                if not s:
                    continue
                yy = y + (j - 1) * 0.22
                if s["n"] > 1:
                    ax.plot([s["min"], s["max"]], [yy, yy], color=color, lw=2, solid_capstyle="round")
                ax.plot(s["mean"], yy, marker=marker, color=color, ms=7, mec=SURFACE, mew=1.5, ls="none")
        ref = summ[t]["fixed"]["logistic"]["mean"]
        ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(labels[::-1], color=TEXT, fontsize=9)
        ax.set_title(TITLES[t], color=TEXT, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("Test AUROC (synthetic data; read the gap, not the level)", color=MUTED, fontsize=9)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="-", lw=2, ms=7, label=lab) for _, lab, c, mk in MODELS]
    handles.append(plt.Line2D([], [], color=MUTED, ls=(0, (3, 3)), lw=1, label="Corrected design, logistic mean"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
