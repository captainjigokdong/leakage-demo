#!/usr/bin/env python3
"""6단계 에이전트용 도구 실행 직전 훅 (PreToolUse). "허용"만 추가한다.

Bash 명령을 ;, &&, ||, |, 줄바꿈, 반복문(for/do/done) 단위로 나눠 모든 조각의 첫 단어가 허용 목록에 있으면 허용한다.
그 밖에는 아무것도 출력하지 않는다 → 기존 권한 규칙(dontAsk + 허용 도구 목록)이 정한다.
허용 규칙만으로는 `echo $?`, `"$f"`처럼 변수를 펼치는 명령이 거부되어 이 훅을 둔다 (2026-10-03 사용자 결정).

판단하기 어려운 형태는 허용하지 않고 기존 규칙에 맡긴다:
명령 치환($(...), 백틱), 하위 셸·묶음((...), {...}), 백그라운드(&), /dev/null이 아닌 파일로 출력 돌리기(> 파일).
파일 삭제·네트워크·권한 변경 명령은 목록에 없다. 금지 경로 접근은 실행기의 오염 검사가 잡는다.
"""
from __future__ import annotations

import json
import re
import shlex
import sys

ALLOWED = frozenset({"python", "python3", "cd", "ls", "cat", "head", "tail", "wc", "echo", "sed", "grep",
                     "zcat", "mkdir", "pwd", "export", "for", "do", "done"})
SEPARATORS = {";", "&&", "||", "|", ";;", "\n"}
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _tokens(cmd: str) -> list[str] | None:
    lx = shlex.shlex(cmd.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lx.whitespace_split = True
    lx.commenters = ""
    try:
        return list(lx)
    except ValueError:          # 닫히지 않은 따옴표 등
        return None


def allowed(cmd: str) -> bool:
    if not cmd.strip() or "$(" in cmd or "`" in cmd or "<<" in cmd:
        return False
    toks = _tokens(cmd)
    if toks is None:
        return False
    segs, cur = [], []
    for i, t in enumerate(toks):
        if t in SEPARATORS:
            segs.append(cur)
            cur = []
            continue
        if t in ("&", "(", ")", "{", "}") or (set(t) <= set("();<>|&") and "&" in t and t not in (">&",)):
            return False
        if t in (">", ">>", ">|", "&>"):
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if nxt != "/dev/null":
                return False
        cur.append(t)
    segs.append(cur)
    for seg in segs:
        words = [w for w in seg]
        while words and ASSIGN.match(words[0]):
            words = words[1:]
        if not words:
            continue
        if words[0] not in ALLOWED:
            return False
        if words[0] == "do" and len(words) > 1 and words[1] not in ALLOWED:
            return False
    return True


def main() -> int:
    try:
        ev = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if ev.get("tool_name") != "Bash":
        return 0
    if allowed(str((ev.get("tool_input") or {}).get("command", ""))):
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                                 "permissionDecisionReason": "허용 목록 명령만으로 이루어짐"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
