"""6단계 실행기 시험. 진짜 에이전트는 부르지 않는다 (가짜 실행기 + 가짜 OS 사용자)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from experiment import run


class FakeUsers(run.OsUser):
    def create(self, name, rd):
        pass

    def wrap(self, name, argv, env):
        return argv

    def remove(self, name, rd):
        pass


def stream(skills, result="보고서\n```findings\n[]\n```", tools_used=(), model=run.MODEL, is_error=False,
           tools=("Bash", "Read", "Skill")):
    ev = [{"type": "system", "subtype": "init", "model": model, "skills": list(skills), "tools": list(tools),
           "mcp_servers": []}]
    for name, inp in tools_used:
        ev.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": inp}]}})
    ev.append({"type": "result", "subtype": "success", "is_error": is_error, "result": result,
               "num_turns": 3, "duration_ms": 1000, "total_cost_usd": 0.5})
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in ev).encode()


def launcher_from(plan):
    """plan: 시도마다 (code, stream bytes) 또는 함수(cwd)->(code, bytes)."""
    it = iter(plan)

    def launch(argv, cwd, timeout):
        step = next(it)
        code, out = step(cwd) if callable(step) else step
        return code, out, b"", 1.0
    return launch


def fake_checker(row, rid):
    return {"report_id": rid, "variant": row["variant"], "exit_code": 0, "checker": {"findings": []}}


def go(tmp_path, cond, plan, rid="abc123def456"):
    row = {"condition": cond, "rep": 1, "variant": run.variants()[0], "order": 0}
    out = tmp_path / "out"
    meta = run.run_row(out, rid, row, run.MODEL, FakeUsers(), launcher_from(plan), tmp_path / "runs", fake_checker)
    return out, meta


def skills(cond):
    return ["leakage-check"] if cond == "가" else []


# ---------------------------------------------------------------- 실행표

def test_schedule_balanced_and_deterministic():
    vs = run.variants()
    assert len(vs) == 20
    s = run.build_schedule(vs, run.REPS, run.SCHEDULE_SEED)
    assert len(s) == 120 and s == run.build_schedule(vs, run.REPS, run.SCHEDULE_SEED)
    for v in vs:
        for c in run.CONDITIONS:
            assert sorted(r["rep"] for r in s.values() if r["variant"] == v and r["condition"] == c) == [1, 2, 3]
    assert sorted(r["order"] for r in s.values()) == list(range(120))
    assert all(len(rid) == 12 for rid in s)


def test_pilot_one_variant_per_type():
    vs = run.pilot_variants()
    assert len(vs) == 2 and {run.design_type(v) for v in vs} == {"dynamic", "fixed"}
    assert len(run.build_schedule(vs, 1, run.PILOT_SEED)) == 4


def test_schedule_is_not_changed(tmp_path):
    s = run.build_schedule(run.variants(), 3, 1)
    run.ensure_schedule(tmp_path, s)
    with pytest.raises(RuntimeError):
        run.ensure_schedule(tmp_path, run.build_schedule(run.variants(), 3, 2))


# ---------------------------------------------------------------- 실행 폴더

def test_run_dirs_differ_only_by_skill(tmp_path):
    v = run.variants()[0]
    a = run.prepare_run(tmp_path, "a", v, "가")
    b = run.prepare_run(tmp_path, "b", v, "나")
    assert (a.ws / ".claude" / "skills" / "leakage-check" / "SKILL.md").exists()
    assert not (b.ws / ".claude").exists() and b.lchome is None
    files = lambda rd: sorted(str(p.relative_to(rd.ws)) for p in rd.ws.rglob("*")
                              if p.is_file() and ".claude" not in p.parts)
    assert files(a) == files(b)
    assert a.inputs == b.inputs
    assert str(run.ROOT) not in json.dumps(run.agent_env(b, "나"))
    assert "LEAKCHECK_HOME" not in run.agent_env(b, "나")


def test_lchome_copy_has_only_checker_files(tmp_path):
    a = run.prepare_run(tmp_path, "a", run.variants()[0], "가")
    assert "leakcheck/checks.py" in a.lchome_files and "designs/schema.json" in a.lchome_files
    for f in a.lchome_files:
        assert not any(f == x or f.startswith(x + "/") for x in run.LEAKCHECK_EXCLUDED), f
        assert not f.endswith(".csv")


def test_lchome_copy_runs_checker(tmp_path):
    import subprocess, sys, os
    a = run.prepare_run(tmp_path, "a", run.variants()[0], "가")
    script = a.ws / ".claude" / "skills" / "leakage-check" / "scripts" / "run_check.py"
    env = {"PATH": os.environ["PATH"], "LEAKCHECK_HOME": str(a.lchome)}
    p = subprocess.run([sys.executable, str(script), str(a.ws / run.variants()[0])], cwd=a.ws, env=env,
                       capture_output=True)
    assert p.returncode in (0, 1), p.stderr.decode()


def test_prompt_same_for_both_conditions():
    argv = run.agent_argv(run.prompt.render("design_X.json", "data"), run.MODEL)
    assert "WebSearch" in argv and "WebFetch" in argv and "--strict-mcp-config" in argv


def test_env_whitelist(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_ADDITIONAL_DIRECTORIES", "/home/user/leakage-demo")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "x")
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "가")
    env = run.agent_env(rd, "가")
    assert "CLAUDE_ADDITIONAL_DIRECTORIES" not in env and "CLAUDE_CODE_SESSION_ID" not in env
    assert env["CLAUDE_CONFIG_DIR"] == str(rd.cfg) and env["HOME"] == str(rd.home)


# ---------------------------------------------------------------- 실행과 재실행

@pytest.mark.parametrize("cond", run.CONDITIONS)
def test_ok_run_saves_blinded_report(tmp_path, cond):
    out, meta = go(tmp_path, cond, [(0, stream(skills(cond)))])
    assert meta["status"] == "done" and meta["retries"] == 0
    rec = json.loads((out / "reports" / "abc123def456.json").read_text())
    assert set(rec) == {"report_id", "variant", "text"}
    assert (out / "checker" / "abc123def456.json").exists() == (cond == "가")
    assert (out / "transcripts" / "abc123def456.jsonl.gz").exists()
    assert not (out / "discarded").exists()


def test_missing_findings_is_not_a_failure(tmp_path):
    out, meta = go(tmp_path, "나", [(0, stream([], result="목록 없는 보고서"))])
    assert meta["status"] == "done" and meta["retries"] == 0
    assert meta["attempts"][0]["findings_block"] is False
    assert json.loads((out / "reports" / "abc123def456.json").read_text())["text"] == "목록 없는 보고서"


@pytest.mark.parametrize("bad,reason", [
    ((1, None), "종료 코드 오류"),
    ((None, None), "시간 초과"),
    ((0, "skills_wrong"), "조작 확인 실패"),
    ((0, "contaminated"), "오염"),
    ((0, "model"), "조작 확인 실패"),
])
def test_mechanical_failure_retried_and_discarded_separately(tmp_path, bad, reason):
    code, kind = bad
    cond = "가"
    if kind == "skills_wrong":
        s = stream([])
    elif kind == "contaminated":
        s = stream(skills(cond), tools_used=[("Bash", {"command": "cat /home/user/leakage-demo/docs/injection_log.md"})])
    elif kind == "model":
        s = stream(skills(cond), model="claude-haiku-4-5")
    else:
        s = stream(skills(cond), result="중간 결과")
    out, meta = go(tmp_path, cond, [(code, s), (0, stream(skills(cond)))])
    assert meta["status"] == "done" and meta["retries"] == 1
    assert any(reason in r for r in meta["retry_reasons"])
    d = out / "discarded" / "abc123def456" / "try1"
    assert (d / "transcript.jsonl.gz").exists() and (d / "attempt.json").exists()
    # 채점기가 읽는 폴더에는 채택된 시도 하나만
    assert [p.name for p in (out / "reports").iterdir()] == ["abc123def456.json"]


def test_all_attempts_fail(tmp_path):
    out, meta = go(tmp_path, "나", [(1, b"")] * (run.MAX_RETRIES + 1))
    assert meta["status"] == "failed" and len(meta["attempts"]) == run.MAX_RETRIES + 1
    assert not (out / "reports").exists()


def test_input_modification_is_contamination(tmp_path):
    def tamper(cwd: Path):
        f = next(cwd.glob("design_*.json"))
        f.chmod(0o644)
        f.write_text("{}")
        return 0, stream([])
    out, meta = go(tmp_path, "나", [tamper, (0, stream([]))])
    assert meta["retries"] == 1 and "오염: 입력 파일 변경" in meta["retry_reasons"]


def test_lchome_path_forbidden_only_for_na(tmp_path):
    rd_a = run.prepare_run(tmp_path, "a", run.variants()[0], "가")
    use = lambda rd: [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash",
                      "input": {"command": f"python {rd.root}/lchome/leakcheck/checks.py"}}]}}]
    assert run.audit(use(rd_a), rd_a, "가")["violations"] == []
    rd_b = run.prepare_run(tmp_path, "b", run.variants()[0], "나")
    assert run.audit(use(rd_b), rd_b, "나")["violations"]
    # 다른 실행 폴더 접근도 금지
    other = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read",
              "input": {"file_path": f"{run.RUN_BASE}/zzz-t1/ws/x"}}]}}]
    assert run.audit(other, rd_b, "나")["violations"]


def test_checker_invocation_recorded_not_required(tmp_path):
    out, meta = go(tmp_path, "가", [(0, stream(skills("가"), tools_used=[
        ("Bash", {"command": "python .claude/skills/leakage-check/scripts/run_check.py design.json --data data"})]))])
    assert meta["attempts"][-1]["audit"]["checker_invoked"] is True
    out2, meta2 = go(tmp_path / "2", "가", [(0, stream(skills("가")))])
    assert meta2["status"] == "done" and meta2["attempts"][-1]["audit"]["checker_invoked"] is False


def test_web_tool_use_is_contamination(tmp_path):
    out, meta = go(tmp_path, "나", [(0, stream([], tools_used=[("WebSearch", {"query": "x"})])), (0, stream([]))])
    assert meta["retries"] == 1


# ---------------------------------------------------------------- 이어 실행, 상태

def test_resume_skips_done(tmp_path):
    vs = run.variants()[:2]
    sched = run.build_schedule(vs, 1, 3)
    out = tmp_path / "out"
    run.ensure_schedule(out, sched)
    calls = []

    def launch(argv, cwd, timeout):
        calls.append(cwd)
        has = (cwd / ".claude").exists()
        return 0, stream(["leakage-check"] if has else []), b"", 1.0

    kw = dict(users=FakeUsers(), launcher=launch, base=tmp_path / "runs", checker=fake_checker)
    run.run_all(out, sched, run.MODEL, workers=2, limit=2, **kw)
    assert len(run.pending(out, sched)) == 2
    run.run_all(out, sched, run.MODEL, workers=2, **kw)
    assert run.pending(out, sched) == [] and len(calls) == 4
    st = run.status(out)
    assert st["가"]["done"] == 2 and st["나"]["done"] == 2 and st["가"]["retries"] == 0


def test_permission_args_same_for_both_and_no_bypass():
    argv = run.agent_argv("x", run.MODEL)
    assert "bypassPermissions" not in " ".join(argv)
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert set(run.ALLOWED_TOOLS) == {"Read", "Glob", "Grep", "Skill", "Write", "Bash(python:*)", "Bash(python3:*)"}


def test_permission_denials_recorded(tmp_path):
    s = stream(skills("가")).replace(b'"total_cost_usd": 0.5}', b'"total_cost_usd": 0.5, "permission_denials": '
        b'[{"tool_name": "Bash", "tool_input": {"command": "cd .claude && python scripts/run_check.py d.json"}},'
        b' {"tool_name": "Edit", "tool_input": {"file_path": "x"}}]}')
    out, meta = go(tmp_path, "가", [(0, s)])
    d = meta["attempts"][-1]["permission_denials"]
    assert d["count"] == 2 and d["checker_denied"] == 1 and d["by_tool"] == {"Bash": 1, "Edit": 1}
    assert meta["status"] == "done"


def test_api_key_passed_and_base_url_dropped(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    env = run.agent_env(rd, "나")
    assert env["ANTHROPIC_API_KEY"] == "test-key" and "ANTHROPIC_BASE_URL" not in env
