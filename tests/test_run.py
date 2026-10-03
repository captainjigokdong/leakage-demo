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
           tools=run.VISIBLE_TOOLS):
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
    # 반복 단위 묶음: 묶음 b의 40행이 순서 40(b-1)..40b-1을 차지
    for b in (1, 2, 3):
        rows = [r for r in s.values() if r["batch"] == b]
        assert len(rows) == 40 and all(r["rep"] == b for r in rows)
        assert sorted(r["order"] for r in rows) == list(range(40 * (b - 1), 40 * b))
        assert {(r["variant"], r["condition"]) for r in rows} == {(v, c) for v in vs for c in run.CONDITIONS}
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


def test_env_drops_only_contaminating_vars(tmp_path):
    base = {"CLAUDE_ADDITIONAL_DIRECTORIES": "/home/user/leakage-demo", "CLAUDE_CODE_SESSION_ID": "x",
            "CLAUDE_CODE_REMOTE_SESSION_ID": "x", "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "1",
            "CLAUDECODE": "1", "SOME_OTHER": "kept", "SSL_CERT_FILE": "/root/.ccr/ca-bundle.crt",
            "LEAKCHECK_HOME": "/home/user/leakage-demo"}
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "가")
    env = run.agent_env(rd, "가", base)
    assert not set(run.ENV_DROP) & set(env)
    assert env["SOME_OTHER"] == "kept"
    assert env["CLAUDE_CONFIG_DIR"] == str(rd.cfg) and env["HOME"] == str(rd.home)
    assert not env["SSL_CERT_FILE"].startswith("/root") and env["LEAKCHECK_HOME"] == str(rd.lchome)
    rb = run.prepare_run(tmp_path, "b", run.variants()[0], "나")
    assert "LEAKCHECK_HOME" not in run.agent_env(rb, "나", base)
    assert "ANTHROPIC_API_KEY" not in open(run.__file__, encoding="utf-8").read()


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
    assert {"Read", "Glob", "Grep", "Skill", "Write", "Bash(python:*)", "Bash(python3:*)",
            "Bash(LEAKCHECK_HOME=*)", "Bash(export:*)"} <= set(run.ALLOWED_TOOLS)
    for c in ("cd", "ls", "cat", "head", "tail", "wc", "echo", "sed", "grep", "zcat", "mkdir", "pwd"):
        assert f"Bash({c}:*)" in run.ALLOWED_TOOLS
    assert not any(x in " ".join(run.ALLOWED_TOOLS) for x in ("rm", "cp", "mv", "curl", "Edit(", "Agent"))
    assert run.MAX_RETRIES == 5


def test_permission_denials_recorded(tmp_path):
    s = stream(skills("가")).replace(b'"total_cost_usd": 0.5}', b'"total_cost_usd": 0.5, "permission_denials": '
        b'[{"tool_name": "Bash", "tool_input": {"command": "cd .claude && python scripts/run_check.py d.json"}},'
        b' {"tool_name": "Edit", "tool_input": {"file_path": "x"}}]}')
    out, meta = go(tmp_path, "가", [(0, s)])
    d = meta["attempts"][-1]["permission_denials"]
    assert d["count"] == 2 and d["checker_denied"] == 1 and d["by_tool"] == {"Bash": 1, "Edit": 1}
    assert meta["status"] == "done"



def test_limit_stops_without_counting_attempt(tmp_path):
    lim = stream([], result="Claude usage limit reached", is_error=True)
    row = {"condition": "나", "rep": 1, "variant": run.variants()[0], "order": 0}
    with pytest.raises(run.LimitReached):
        run.run_row(tmp_path / "out", "abc123def456", row, run.MODEL, FakeUsers(), launcher_from([(1, lim)]),
                    tmp_path / "runs", fake_checker)
    assert not (tmp_path / "out" / "meta").exists()
    assert list((tmp_path / "out" / "discarded" / "abc123def456").iterdir())


def test_report_mentioning_limit_is_not_limit():
    assert not run.is_limit({"is_error": False, "result": "rate limit 관련 특징 ..."}, b"")


def test_batches_resume_after_limit(tmp_path):
    vs = run.variants()[:2]
    sched = run.build_schedule(vs, 2, 3)        # 묶음 2개 × 4행
    out = tmp_path / "out"
    run.ensure_schedule(out, sched)
    n = {"calls": 0}

    def launch(argv, cwd, timeout):
        n["calls"] += 1
        if n["calls"] == 6:
            return 1, stream([], result="usage limit reached", is_error=True), b"", 1.0
        has = (cwd / ".claude").exists()
        return 0, stream(["leakage-check"] if has else []), b"", 1.0

    kw = dict(users=FakeUsers(), launcher=launch, base=tmp_path / "runs", checker=fake_checker)
    assert run.run_batches(out, sched, run.MODEL, 1, commit=False, **kw) is False
    first = {rid for rid, r in sched.items() if r["batch"] == 1}
    assert not set(run.pending(out, sched)) & first           # 묶음 1은 끝남
    assert len(run.pending(out, sched)) == 3                   # 묶음 2에서 1개만 끝나고 멈춤
    assert run.run_batches(out, sched, run.MODEL, 1, commit=False, **kw) is True
    assert run.pending(out, sched) == []


# ---------------------------------------------------------------- root 실행: 권한 규칙과 넓힌 오염 검사

def test_root_mode_and_deny_settings():
    assert run.RUN_AS == "root" and isinstance(run.default_users(), run.RootUser)
    rules = run.deny_settings()["permissions"]["deny"]
    for p in ("/home/user", "/root", "/mnt/user-data"):
        for t in ("Read", "Glob", "Grep"):
            assert f"{t}(/{p}/**)" in rules
    argv = run.RootUser().wrap("x", run.agent_argv("t", run.MODEL, Path("/s.json")), {"A": "1"})
    assert argv[:3] == ["env", "-i", "A=1"] and "--settings" in argv and "setpriv" not in argv
    i = argv.index("--tools")
    assert argv[i + 1:i + 7] == list(run.VISIBLE_TOOLS) and "Agent" not in run.VISIBLE_TOOLS


@pytest.mark.parametrize("cmd", [
    "python3 -c \"import os; print(os.listdir('/'))\"",
    "python -c \"import os; [print(r) for r in os.walk('/home')]\"",
    "python3 -c \"open('../../home/x')\"",
    "python3 script.py ../..",
    "python3 -c \"import glob; print(glob.glob('/srv/*'))\"",
    "python3 -c \"print(open('/root/.claude/x').read())\"",
    "python3 -c \"import os; os.path.expanduser('~root')\"",
    "python3 -c \"from pathlib import Path; print(list(Path('/mnt').iterdir()))\"",
])
def test_broadened_audit_flags(tmp_path, cmd):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    ev = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash",
          "input": {"command": cmd}}]}}]
    assert run.audit(ev, rd, "나")["violations"], cmd


@pytest.mark.parametrize("cmd", [
    "python3 -c \"import pandas as pd; df = pd.read_csv('data/labs.csv.gz'); print(df.x / 2)\"",
    "python .claude/skills/leakage-check/scripts/run_check.py design_0B14.json --data data",
    "python3 -c \"print(1/3, 'a/b')\"",
    "python3 -c \"x = [1, 2]; print(x[...])\"",
])
def test_broadened_audit_allows_normal_work(tmp_path, cmd):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "가")
    ev = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash",
          "input": {"command": cmd}}]}}]
    assert run.audit(ev, rd, "가")["violations"] == [], cmd


def test_written_script_content_audited(tmp_path):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    ev = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Write",
          "input": {"file_path": f"{rd.ws}/s.py", "content": "import os\nfor r in os.walk('/'):\n    pass\n"}}]}}]
    assert run.audit(ev, rd, "나")["violations"]
    (rd.ws / "t.py").write_text("print(open('/home/user/leakage-demo/CLAUDE.md').read())\n")
    assert run.audit_scripts(rd, "나")
    (rd.ws / "t.py").write_text("import pandas\n")
    assert run.audit_scripts(rd, "나") == []


def test_contamination_discards_counted(tmp_path):
    bad = stream([], tools_used=[("Bash", {"command": "python3 -c \"import os; os.listdir('/')\""})])
    row = {"condition": "나", "rep": 1, "batch": 1, "variant": run.variants()[0], "order": 0}
    out = tmp_path / "out"
    run.ensure_schedule(out, {"abc123def456": row})
    run.run_row(out, "abc123def456", row, run.MODEL, FakeUsers(), launcher_from([(0, bad), (0, stream([]))]),
                tmp_path / "runs", fake_checker)
    st = run.status(out)
    assert st["나"]["contamination_discarded"] == 1 and st["나"]["done"] == 1


def test_auth_error_stops_without_retry(tmp_path):
    bad = stream([], result="Authentication error · This may be a temporary network issue", is_error=True)
    row = {"condition": "나", "rep": 1, "variant": run.variants()[0], "order": 0}
    with pytest.raises(run.LimitReached):
        run.run_row(tmp_path / "out", "abc123def456", row, run.MODEL, FakeUsers(), launcher_from([(1, bad)]),
                    tmp_path / "runs", fake_checker)



# ---------------------------------------------------------------- 시험 실행 3차 폐기 6건 재검사 (2026-10-03)

ARCHIVE = run.ROOT / "experiment" / "pilot" / "_run3_20261003" / "discarded"
# 오탐 4건: 자기 임시 폴더(/tmp/claude-0/-srv-leakruns-<자기 실행>-ws/...), 코드 안의 '/' 문자열 (split('/'))
FALSE_POS = [("278c665bce74", 2), ("8dab1848f9d0", 1), ("8dab1848f9d0", 2), ("8dab1848f9d0", 3)]
# 실제 탐색 2건: 하위 에이전트에게 find / 지시, find /srv/leakruns (다른 실행 폴더 순회) → 계속 오염
TRUE_POS = [("278c665bce74", 1), ("278c665bce74", 3)]
COND = {"278c665bce74": "가", "8dab1848f9d0": "나"}


def _reaudit(rid, k):
    import gzip as _gz
    ev = run.parse_stream(_gz.decompress((ARCHIVE / rid / f"try{k}" / "transcript.jsonl.gz").read_bytes()))
    root = run.RUN_BASE / f"{rid}-t{k}"
    rd = run.RunDir(root, root / "ws", root / "home", root / "cfg", root / "lchome" if COND[rid] == "가" else None)
    return run.audit(ev, rd, COND[rid])["violations"]


@pytest.mark.skipif(not ARCHIVE.exists(), reason="보관 기록 없음")
@pytest.mark.parametrize("rid,k", FALSE_POS)
def test_pilot3_false_positives_cleared(rid, k):
    assert _reaudit(rid, k) == []


@pytest.mark.skipif(not ARCHIVE.exists(), reason="보관 기록 없음")
@pytest.mark.parametrize("rid,k", TRUE_POS)
def test_pilot3_real_exploration_still_flagged(rid, k):
    assert _reaudit(rid, k)


def test_own_tmp_allowed_other_tmp_forbidden(tmp_path):
    rd = run.prepare_run(tmp_path, "abc123def456-t2", run.variants()[0], "나")
    use = lambda p: [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read",
                      "input": {"file_path": p}}]}}]
    assert run.audit(use("/tmp/claude-0/-srv-leakruns-abc123def456-t2-ws/x/y.py"), rd, "나")["violations"] == []
    for bad in ("/tmp/claude-0/-srv-leakruns-abc123def456-t1-ws/x", "/tmp/claude-0/-home-user-leakage-demo/x",
                "/tmp/claude-0"):
        assert run.audit(use(bad), rd, "나")["violations"], bad


@pytest.mark.parametrize("cmd", ["ls /", "find / -name x", "python3 -c \"import os; os.listdir('/')\"",
                                 "python3 -c \"import os; list(os.walk('/'))\"", "ls -la /srv",
                                 "python3 -c \"from pathlib import Path; list(Path('/').rglob('*'))\""])
def test_listing_commands_flagged(tmp_path, cmd):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    ev = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}]
    assert run.audit(ev, rd, "나")["violations"], cmd


@pytest.mark.parametrize("cmd", ["python3 -c \"print('a/b'.split('/'))\"", "python3 -c \"x = '/'.join(['a', 'b'])\"",
                                 "python3 -c \"import os; os.listdir('data')\""])
def test_slash_string_not_flagged(tmp_path, cmd):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    ev = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}]
    assert run.audit(ev, rd, "나")["violations"] == [], cmd


def test_checker_invoked_ignores_denied_calls(tmp_path):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "가")
    ev = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "id": "t1",
           "input": {"command": "python3 .claude/skills/leakage-check/scripts/run_check.py d.json --data data; echo $?"}}]}},
          {"type": "result", "permission_denials": [{"tool_use_id": "t1", "tool_name": "Bash", "tool_input": {}}]}]
    assert run.audit(ev, rd, "가")["checker_invoked"] is False


def test_visible_tools_manipulation_check():
    ok = {"model": run.MODEL, "skills": [], "tools": list(run.VISIBLE_TOOLS), "mcp_servers": []}
    assert run.manipulation_check(ok, "나", run.MODEL) == []
    assert run.manipulation_check({**ok, "tools": [*run.VISIBLE_TOOLS, "Agent"]}, "나", run.MODEL)


def test_tmpdir_inside_run_root(tmp_path):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    assert run.agent_env(rd, "나", {})["TMPDIR"] == str(rd.root / "tmp") and (rd.root / "tmp").is_dir()



# ---------------------------------------------------------------- 시험 실행 4차 폐기 4건 재검사 (2026-10-03)

ARCHIVE4 = run.ROOT / "experiment" / "pilot" / "_run4_20261003" / "discarded"
# 4건 모두 오탐: 점검기 결과를 ws/../tmp/... (자기 실행 폴더 안 임시 폴더)에 저장 3건, 경로 없는 `ls -R` 1건
FALSE_POS4 = [("1c9b176c645e", 1), ("278c665bce74", 1), ("278c665bce74", 2), ("278c665bce74", 3)]


@pytest.mark.skipif(not ARCHIVE4.exists(), reason="보관 기록 없음")
@pytest.mark.parametrize("rid,k", FALSE_POS4)
def test_pilot4_false_positives_cleared(rid, k):
    import gzip as _gz
    ev = run.parse_stream(_gz.decompress((ARCHIVE4 / rid / f"try{k}" / "transcript.jsonl.gz").read_bytes()))
    root = run.RUN_BASE / f"{rid}-t{k}"
    rd = run.RunDir(root, root / "ws", root / "home", root / "cfg", root / "lchome")
    assert run.audit(ev, rd, "가")["violations"] == []


@pytest.mark.parametrize("cmd,bad", [
    ("python3 x.py --json ../tmp/claude-0/a/scratchpad/r.json", False),
    ("ls -R", False),
    ("ls .claude/skills/leakage-check -R", False),
    ("python3 -c \"open('../../other-t1/ws/x')\"", True),
    ("cd .. && python3 x.py", True),
    ("python3 -c \"print(open('../lchome/leakcheck/rules.py').read())\"", True),   # (나)에서 사본 경로
    ("ls -R /", True),
])
def test_dotdot_and_listing_rules(tmp_path, cmd, bad):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    ev = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}]
    assert bool(run.audit(ev, rd, "나")["violations"]) is bad, cmd



def test_settings_allow_own_run_only(tmp_path):
    rd = run.prepare_run(tmp_path, "a", run.variants()[0], "나")
    perms = run.deny_settings(rd)["permissions"]
    assert perms["allow"] == [f"Edit(/{rd.root}/**)"]
    assert "Read(//home/user/**)" in perms["deny"]


def test_blank_rows_counted(tmp_path):
    row = {"condition": "가", "rep": 1, "batch": 1, "variant": run.variants()[0], "order": 0}
    out = tmp_path / "out"
    run.ensure_schedule(out, {"abc123def456": row})
    run.run_row(out, "abc123def456", row, run.MODEL, FakeUsers(), launcher_from([(1, b"")] * 6),
                tmp_path / "runs", fake_checker)
    st = run.status(out)
    assert st["가"]["blank"] == 1 and st["가"]["retries"] == 5


def test_no_checker_after_denial_recorded(tmp_path):
    s = stream(skills("가"), tools_used=[("Bash", {"command": "ls"})]).replace(
        b'"total_cost_usd": 0.5}', b'"total_cost_usd": 0.5, "permission_denials": [{"tool_name": "Bash", "tool_input": {"command": "x"}}]}')
    out, meta = go(tmp_path, "가", [(0, s)])
    assert meta["attempts"][-1]["no_checker_after_denial"] is True
    out2, meta2 = go(tmp_path / "2", "나", [(0, s.replace(b'leakage-check', b'zz'))])
    assert meta2["attempts"][-1]["no_checker_after_denial"] is False



def test_discard_report_and_first_attempts(tmp_path):
    bad = stream([], result="첫 시도 보고서", tools_used=[("Bash", {"command": "find / -name x"})])
    row = {"condition": "나", "rep": 1, "batch": 1, "variant": run.variants()[0], "order": 0}
    row2 = {**row, "order": 1}
    out = tmp_path / "out"
    run.ensure_schedule(out, {"abc123def456": row, "bbb123def456": row2})
    run.run_row(out, "abc123def456", row, run.MODEL, FakeUsers(), launcher_from([(0, bad), (0, stream([]))]),
                tmp_path / "runs", fake_checker)
    run.run_row(out, "bbb123def456", row2, run.MODEL, FakeUsers(), launcher_from([(0, stream([]))]),
                tmp_path / "runs", fake_checker)
    rep = run.discard_report(out)["묶음1/나"]
    assert rep["discards"] == 1 and rep["전체 탐색 시도"] == 1 and rep["report_text_kept"] == 1
    assert len(rep["executed_flagged_calls"]) == 1          # 가짜 기록에는 거부 정보가 없으니 실행된 것으로 셈
    n = run.export_first_attempts(out, tmp_path / "first")
    assert n == {"from_discarded": 1, "from_reports": 1, "missing": 0}
    rec = json.loads((tmp_path / "first" / "abc123def456.json").read_text())
    assert set(rec) == {"report_id", "variant", "text"} and rec["text"] == "첫 시도 보고서"
