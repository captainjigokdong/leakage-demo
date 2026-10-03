"""에이전트 지시문 (5단계에 고정). (가)/(나) 두 조건에 똑같이 쓴다.

지시문 파일이 바뀌면 PROMPT_SHA256과 달라져 render()가 멈춘다. 값은 docs/success_criteria.md에도 적혀 있다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PROMPT_FILE = Path(__file__).resolve().parent / "agent_prompt.md"
PROMPT_SHA256 = "5e444b22ff4c6ced840de3cf5f07ed1583c3a6bc47e608093ea028592ef29cb8"


def text() -> str:
    t = PROMPT_FILE.read_text(encoding="utf-8")
    h = hashlib.sha256(t.encode("utf-8")).hexdigest()
    if h != PROMPT_SHA256:
        raise RuntimeError(f"지시문이 고정본과 다르다 (sha256 {h}). 5단계 이후 지시문은 바꾸지 않는다.")
    return t


def render(design_file: str, data_dir: str) -> str:
    """조건과 무관하게 설계서·데이터 경로만 채운다."""
    return text().replace("{{DESIGN_FILE}}", design_file).replace("{{DATA_DIR}}", data_dir)
