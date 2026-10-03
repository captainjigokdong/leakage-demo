"""6단계 도구 실행 직전 훅 시험: 허용 목록 명령만으로 된 Bash 명령만 "허용"하고, 그 밖에는 말하지 않는다."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from experiment import bash_allow_hook as hook
from experiment import run

S = "python3 .claude/skills/leakage-check/scripts/run_check.py d.json --data data"


@pytest.mark.parametrize("cmd", [
    S,
    f"cd /srv/leakruns/x/ws; {S} --json ../tmp/r.json; echo EXIT $?; cat data/MANIFEST.json",
    f"{S} | tail -1; echo EXIT=$?",
    f"LEAKCHECK_HOME=/srv/leakruns/x/lchome {S} | head -5",
    f"export LEAKCHECK_HOME=/x; {S} 2>&1 | sed -n '1,3p'",
    'cd ws; for f in data/*; do echo "== $f"; head -3 "$f"; wc -l < "$f"; done',
    'for f in *.gz; do echo "== $f"; zcat $f | head -4; done',
    'python3 -c "import pandas as pd\nprint(pd.read_csv(\'data/a.csv.gz\').shape); x = 1; y = 2"',
    f"{S} > /dev/null && echo ok || echo fail",
    "mkdir -p out && ls && pwd",
    "grep -n leakage .claude/skills/leakage-check/SKILL.md",
])
def test_allowed(cmd):
    assert hook.allowed(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "rm -rf data", "cd ws; rm x", "curl http://x", "wget x", "chmod 777 x", "chown a x", "cp a b", "mv a b",
    f"{S}; rm out.txt", "python3 x.py & sleep 1", "echo $(rm -rf /)", "echo `id`", "(cd /; ls)",
    "find / -name x", "xargs rm", "bash -c 'ls'", "sh x.sh", "env python3 x.py",
    f"{S} > out.txt", "echo x >> log", "python3 - <<'EOF'\nprint(1)\nEOF",
    "for f in *; do rm $f; done", "ls; sudo ls", "", "echo 'unclosed",
])
def test_not_allowed(cmd):
    assert not hook.allowed(cmd), cmd


def _call(payload: dict) -> str:
    p = subprocess.run([sys.executable, str(run.HOOK_SRC)], input=json.dumps(payload), capture_output=True, text=True)
    assert p.returncode == 0
    return p.stdout.strip()


def test_hook_protocol():
    out = json.loads(_call({"tool_name": "Bash", "tool_input": {"command": f"{S}; echo $?"}}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert _call({"tool_name": "Bash", "tool_input": {"command": "rm -rf x"}}) == ""   # 말하지 않음 → 기존 규칙
    assert _call({"tool_name": "Read", "tool_input": {"file_path": "/home/user/x"}}) == ""


def test_hook_only_allows_never_denies():
    src = run.HOOK_SRC.read_text(encoding="utf-8")
    assert '"deny"' not in src and '"ask"' not in src


def test_settings_have_hook_for_both_conditions(tmp_path):
    for cond in run.CONDITIONS:
        rd = run.prepare_run(tmp_path, f"r{cond}", run.variants()[0], cond)
        st = run.deny_settings(rd)
        h = st["hooks"]["PreToolUse"][0]
        assert h["matcher"] == "Bash" and str(rd.root / "bash_allow_hook.py") in h["hooks"][0]["command"]
