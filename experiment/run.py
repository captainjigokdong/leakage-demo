"""6단계 (가)/(나) 실행기.

변형 × 조건 × 반복마다 저장소 밖에 격리된 실행 폴더를 만들고, 같은 지시문·같은 모델로 에이전트(Claude Code 헤드리스)를 돌린다.

    python -m experiment.run schedule              # 실행표 만들기 (이미 있으면 같은지 확인)
    python -m experiment.run preflight             # 격리 확인 (실행 사용자가 저장소를 못 읽는지)
    python -m experiment.run pilot                 # 작은 시험 실행: 변형 2개 × 조건 2 × 1회 → experiment/pilot/
    python -m experiment.run main [--limit N]      # 본 실행 120회 = 묶음 3개(반복 단위, 40회씩) → experiment/runs/
                                                   # 묶음마다 커밋·푸시. 한도에 걸리면 멈추고, 다시 부르면 이어 간다
    python -m experiment.run status [--out DIR]    # 진행 상황, 조건별 재실행 횟수·사유

저장 (DIR = experiment/runs 또는 experiment/pilot)
- DIR/reports/<id>.json      {"report_id", "variant", "text"}  ← 채점 대상. 조건 표시 없음
- DIR/checker/<id>.json      {"report_id", "variant", "exit_code", "checker"}  ← (가)만
- DIR/conditions.json        {id: {"condition", "rep", "variant", "order"}}  ← 조건은 여기에만
- DIR/meta/<id>.json         실행 기록 요약 (시도별 상태·사유, 시간, 사용량, 스킬 목록, 점검기 호출 여부, 접근 경로 검사)
- DIR/transcripts/<id>.jsonl.gz   채택된 시도의 실행 기록 (stream-json)
- DIR/discarded/<id>/try<k>/      버린 시도 (기계적 실패만): 실행 기록, 결과 문장, 사유. 채점 대상과 분리

재실행은 기계적 실패에만 한다: 종료 코드 오류, 시간 초과, 오염(허용되지 않은 경로·도구 접근, 입력 파일 변경), 조작 확인 실패.
보고서에 findings 블록이 없는 것은 실패가 아니다. 그대로 저장해 채점에 넘긴다 (채점 규칙의 형식 오류로 처리).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from experiment import prompt

ROOT = Path(__file__).resolve().parent.parent
VARIANT_DIR = ROOT / "designs" / "variants"
DATA_DIR = ROOT / "data" / "synth"
SKILL_SRC = ROOT / "skill_src" / "leakage-check"
RUNS_OUT = ROOT / "experiment" / "runs"
PILOT_OUT = ROOT / "experiment" / "pilot"

SCHEDULE_SEED = 20261004
PILOT_SEED = 20261005
REPS = 3                   # 반복 = 묶음. 묶음 하나 = 변형 20 × 조건 2 = 40회
CONDITIONS = ("가", "나")
MODEL = "claude-opus-5-5"
TIMEOUT_S = 20 * 60
MAX_RETRIES = 2            # 기계적 실패 시 다시 실행하는 최대 횟수 (시도는 최대 3회)
WORKERS = 3

RUN_BASE = Path("/srv/leakruns")       # 실행 폴더 (실행마다 다른 OS 사용자, 0700)
SHARED = Path("/srv/leakshared")       # 실행 사용자가 읽을 수 있는 공용 파일 (CA 묶음만)
CA_SRC = Path("/root/.ccr/ca-bundle.crt")

# 두 조건에 똑같이 준다. 웹 검색·웹 가져오기는 둘 다 끈다.
DISALLOWED_TOOLS = ("WebSearch", "WebFetch")
# 권한 방식 (2026-10-03 사용자 결정): bypassPermissions는 쓰지 않는다. dontAsk 모드에서 미리 허용한 도구만 쓰고
# 그 밖의 호출은 거부된다. 두 조건에 같은 값. python과 python3를 둘 다 허용한다 (SKILL.md가 python으로 부른다).
ALLOWED_TOOLS = ("Read", "Glob", "Grep", "Skill", "Write", "Bash(python:*)", "Bash(python3:*)")
PERMISSION_ARGS: tuple[str, ...] = ("--permission-mode", "dontAsk", "--allowedTools", *ALLOWED_TOOLS)

# (가) LEAKCHECK_HOME 사본: 점검에 필요한 파일만. tests/, designs/inject.py, 오류 목록 CSV, docs/는 넣지 않는다.
LEAKCHECK_FILES = (
    *(f"leakcheck/{p.name}" for p in sorted((ROOT / "leakcheck").glob("*.py"))),
    "synth/__init__.py", "synth/generate.py", "synth/stats.py",
    "designs/schema.json",
)
LEAKCHECK_EXCLUDED = ("tests", "docs", "designs/inject.py", "designs/error_catalog_public.csv",
                      "designs/variants", "designs/base", "sealed", "experiment", "CLAUDE.md")

# 실행 사용자 환경 (2026-10-03 사용자 결정): 세션 환경을 그대로 넘기되, 오염을 일으키는 것으로 확인된 변수만 뺀다.
# 인증 변수가 무엇인지 찾거나 값을 보지 않는다. 확인된 오염: 이 세션과 같은 세션 id로 뜸(0단계), 저장소를 추가 폴더로 넣음.
ENV_DROP = ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID",
            "CLAUDE_ADDITIONAL_DIRECTORIES", "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD")
# 실행 폴더별로 덮어쓰는 변수 (집·설정 폴더·PATH, 실행 사용자가 못 읽는 /root 안 CA 경로)
CA_VARS = ("NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT",
           "AWS_CA_BUNDLE", "NIX_SSL_CERT_FILE", "DENO_CERT", "CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE", "HEX_CACERTS_PATH")

# 실행 기록에 나타나면 오염으로 보는 경로 (실행 폴더 밖 민감 경로)
FORBIDDEN_PREFIXES = (str(ROOT), "/home/user", "/root", "/mnt/user-data", "/tmp/claude-0")
# 실행 사용자에게 막아 두는 폴더 (preflight가 o-rwx로 바꾼다. root는 영향 없음)
HARDEN_DIRS = (Path("/home/user"), Path("/mnt/user-data"))


# ---------------------------------------------------------------- 실행표

def variants() -> list[str]:
    return sorted(p.name for p in VARIANT_DIR.glob("design_*.json"))


def design_type(v: str) -> str:
    return json.loads((VARIANT_DIR / v).read_text(encoding="utf-8"))["design_type"]


def build_schedule(vs: list[str], reps: int, seed: int) -> dict[str, dict]:
    """{report_id: {condition, rep, batch, variant, order}}.

    반복 단위 묶음: 묶음 r = 반복 r의 (변형 × 조건) 전부. 묶음 1을 다 돌린 뒤 묶음 2로 간다.
    묶음 안에서는 순서를 섞어 조건·변형이 시간에 몰리지 않게 한다.
    """
    rng = random.Random(seed)
    out, used, order = {}, set(), 0
    for r in range(1, reps + 1):
        rows = [(v, c) for v in vs for c in CONDITIONS]
        rng.shuffle(rows)
        for v, c in rows:
            rid = "%012x" % rng.getrandbits(48)
            while rid in used:
                rid = "%012x" % rng.getrandbits(48)
            used.add(rid)
            out[rid] = {"condition": c, "rep": r, "batch": r, "variant": v, "order": order}
            order += 1
    return out


def pilot_variants(seed: int = PILOT_SEED) -> list[str]:
    """유형별 1개씩 추첨 (결함 여부는 모른다)."""
    rng = random.Random(seed)
    by: dict[str, list[str]] = {}
    for v in variants():
        by.setdefault(design_type(v), []).append(v)
    return [rng.choice(sorted(by[t])) for t in sorted(by)]


def ensure_schedule(out: Path, sched: dict[str, dict]) -> dict[str, dict]:
    f = out / "conditions.json"
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old != sched:
            raise RuntimeError(f"{f}가 지금 만든 실행표와 다르다. 실행표는 바꾸지 않는다.")
        return old
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sched, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return sched


# ---------------------------------------------------------------- 실행 폴더

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass
class RunDir:
    root: Path
    ws: Path
    home: Path
    cfg: Path
    lchome: Path | None
    inputs: dict[str, str] = field(default_factory=dict)     # ws 상대 경로 → sha256
    lchome_files: list[str] = field(default_factory=list)


def prepare_run(base: Path, run_name: str, variant: str, condition: str) -> RunDir:
    root = base / run_name
    if root.exists():
        shutil.rmtree(root)
    ws, home, cfg = root / "ws", root / "home", root / "cfg"
    for d in (ws, home, cfg, ws / "data"):
        d.mkdir(parents=True)
    rd = RunDir(root, ws, home, cfg, None)
    shutil.copy2(VARIANT_DIR / variant, ws / variant)
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file():
            shutil.copy2(p, ws / "data" / p.name)
    for p in [ws / variant, *sorted((ws / "data").iterdir())]:
        p.chmod(0o444)
        rd.inputs[str(p.relative_to(ws))] = sha256(p)
    if condition == "가":
        shutil.copytree(SKILL_SRC, ws / ".claude" / "skills" / "leakage-check")
        rd.lchome = root / "lchome"
        for rel in LEAKCHECK_FILES:
            dst = rd.lchome / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)
        rd.lchome_files = sorted(str(p.relative_to(rd.lchome)) for p in rd.lchome.rglob("*") if p.is_file())
    return rd


def inputs_changed(rd: RunDir) -> list[str]:
    return [rel for rel, h in rd.inputs.items() if not (rd.ws / rel).exists() or sha256(rd.ws / rel) != h]


# ---------------------------------------------------------------- 에이전트 실행

def agent_env(rd: RunDir, condition: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in (os.environ if base_env is None else base_env).items() if k not in ENV_DROP}
    ca = str(SHARED / "ca-bundle.crt")
    env.update({k: ca for k in CA_VARS if k in env or k in ("NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE")})
    env.update({"PATH": "/opt/node22/bin:/usr/local/bin:/usr/bin:/bin", "HOME": str(rd.home),
                "CLAUDE_CONFIG_DIR": str(rd.cfg)})
    env.pop("LEAKCHECK_HOME", None)
    if condition == "가":
        env["LEAKCHECK_HOME"] = str(rd.lchome)
    return env


def agent_argv(text: str, model: str) -> list[str]:
    return ["claude", "-p", text, "--model", model, "--output-format", "stream-json", "--verbose",
            "--strict-mcp-config", "--disallowedTools", *DISALLOWED_TOOLS, *PERMISSION_ARGS]


class OsUser:
    """실행마다 새 OS 사용자. 실행 폴더는 그 사용자만 읽는다 (0700)."""

    def create(self, name: str, rd: RunDir) -> None:
        subprocess.run(["useradd", "-M", "-d", str(rd.home), "-s", "/usr/sbin/nologin", name], check=True)
        subprocess.run(["chown", "-R", f"{name}:{name}", str(rd.root)], check=True)
        rd.root.chmod(0o700)

    def wrap(self, name: str, argv: list[str], env: dict[str, str]) -> list[str]:
        return ["setpriv", f"--reuid={name}", f"--regid={name}", "--clear-groups",
                "env", "-i", *(f"{k}={v}" for k, v in env.items()), *argv]

    def remove(self, name: str, rd: RunDir) -> None:
        subprocess.run(["chown", "-R", "root:root", str(rd.root)], check=False)
        subprocess.run(["userdel", name], check=False, capture_output=True)


def launch(argv: list[str], cwd: Path, timeout: int) -> tuple[int | None, bytes, bytes, float]:
    """(종료 코드 또는 시간 초과 None, stdout, stderr, 초)."""
    t0 = time.monotonic()
    p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, time.monotonic() - t0
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return None, out, err, time.monotonic() - t0


# ---------------------------------------------------------------- 실행 기록 분석

PATH_RE = re.compile(r"(?<![\w.~$])(/[A-Za-z0-9_.\-/]+)")


def parse_stream(raw: bytes) -> list[dict]:
    ev = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return ev


def tool_uses(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        if e.get("type") != "assistant":
            continue
        for c in (e.get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                out.append({"name": c.get("name"), "input": c.get("input") or {}})
    return out


def _strings(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def audit(events: list[dict], rd: RunDir, condition: str) -> dict:
    """도구 호출 입력에 나타난 경로·도구를 검사한다 (오염 = 허용되지 않은 접근 시도. 실패한 시도도 센다)."""
    uses = tool_uses(events)
    own = str(rd.root)
    forbidden = [*FORBIDDEN_PREFIXES, str(RUN_BASE)]
    hits = []
    for u in uses:
        if u["name"] in DISALLOWED_TOOLS:
            hits.append({"tool": u["name"], "path": None, "why": "꺼 둔 도구"})
        for s in _strings(u["input"]):
            for m in PATH_RE.findall(s):
                p = m.rstrip(".")
                inside = p == own or p.startswith(own + "/")
                if inside and condition != "가" and (p + "/").startswith(own + "/lchome/"):
                    hits.append({"tool": u["name"], "path": p, "why": "(나)에서 점검기 사본 경로"})
                elif not inside and any(p == f or p.startswith(f + "/") for f in forbidden):
                    hits.append({"tool": u["name"], "path": p, "why": "실행 폴더 밖 금지 경로"})
    bash = [u["input"].get("command", "") for u in uses if u["name"] == "Bash"]
    return {
        "tool_counts": {n: sum(u["name"] == n for u in uses) for n in sorted({u["name"] for u in uses})},
        "checker_invoked": any("run_check.py" in c for c in bash),
        "skill_tool_invoked": any(u["name"] == "Skill" and "leakage-check" in json.dumps(u["input"], ensure_ascii=False)
                                  for u in uses),
        "skill_md_read": any(u["name"] == "Read" and str(u["input"].get("file_path", "")).endswith("SKILL.md")
                             for u in uses),
        "violations": hits,
    }


def init_info(events: list[dict]) -> dict:
    for e in events:
        if e.get("type") == "system" and e.get("subtype") == "init":
            return {"model": e.get("model"), "skills": sorted(e.get("skills") or []),
                    "tools": sorted(e.get("tools") or []), "mcp_servers": e.get("mcp_servers") or [],
                    "cwd": e.get("cwd"), "permission_mode": e.get("permissionMode")}
    return {}


def result_info(events: list[dict]) -> dict:
    for e in reversed(events):
        if e.get("type") == "result":
            return {k: e.get(k) for k in ("subtype", "is_error", "result", "num_turns", "duration_ms",
                                          "total_cost_usd", "usage", "modelUsage", "terminal_reason",
                                          "permission_denials", "api_error_status")}
    return {}


LIMIT_RE = re.compile(r"usage limit|rate limit|limit reached|429|overloaded|too many requests", re.I)


def is_limit(res: dict, err: bytes) -> bool:
    """요금제 한도·속도 제한. 오류 문장만 본다 (오류가 아닌 보고서 본문은 보지 않는다)."""
    if res.get("api_error_status") in (429, 529):
        return True
    if res and not res.get("is_error"):
        return False
    return bool(LIMIT_RE.search((res.get("result") or "") + err.decode("utf-8", "replace")[-2000:]))


def denials(res: dict) -> dict:
    """권한이 거부된 도구 호출 수 (점검기 호출이 거부된 수 따로)."""
    d = res.get("permission_denials") or []
    return {"count": len(d), "by_tool": {n: sum(x.get("tool_name") == n for x in d) for n in sorted({x.get("tool_name") for x in d})},
            "checker_denied": sum("run_check.py" in json.dumps(x.get("tool_input") or {}, ensure_ascii=False) for x in d),
            "commands": [str((x.get("tool_input") or {}).get("command", x.get("tool_name")))[:200] for x in d]}


def manipulation_check(init: dict, condition: str, model: str) -> list[str]:
    bad = []
    if not init:
        return ["init 이벤트 없음"]
    if init.get("model") != model:
        bad.append(f"모델 불일치: {init.get('model')}")
    has = "leakage-check" in init.get("skills", [])
    if condition == "가" and not has:
        bad.append("(가)에 leakage-check 스킬이 로드되지 않음")
    if condition == "나" and has:
        bad.append("(나)에 leakage-check 스킬이 로드됨")
    if set(DISALLOWED_TOOLS) & set(init.get("tools", [])):
        bad.append("웹 도구가 켜져 있음")
    if init.get("mcp_servers"):
        bad.append("MCP 서버가 연결됨")
    return bad


# ---------------------------------------------------------------- 한 번 실행

@dataclass
class Attempt:
    status: str                    # "ok" | "fail"
    reasons: list[str]
    exit_code: int | None
    seconds: float
    events: list[dict]
    raw: bytes
    stderr: bytes
    meta: dict


def run_once(rid: str, row: dict, try_no: int, model: str, users: OsUser, launcher=launch,
             base: Path = RUN_BASE) -> Attempt:
    name = f"lr{rid[:10]}t{try_no}"
    rd = prepare_run(base, f"{rid}-t{try_no}", row["variant"], row["condition"])
    text = prompt.render(row["variant"], "data")
    users.create(name, rd)
    try:
        argv = users.wrap(name, agent_argv(text, model), agent_env(rd, row["condition"]))
        code, out, err, secs = launcher(argv, rd.ws, TIMEOUT_S)
    finally:
        users.remove(name, rd)
    events = parse_stream(out)
    init, res = init_info(events), result_info(events)
    aud = audit(events, rd, row["condition"])
    changed = inputs_changed(rd)
    reasons = []
    if code is None:
        reasons.append("시간 초과")
    elif code != 0 or res.get("is_error") or not res:
        reasons.append(f"종료 코드 오류 (code={code}, subtype={res.get('subtype')})")
    if aud["violations"]:
        reasons.append("오염: 허용되지 않은 접근")
    if changed:
        reasons.append("오염: 입력 파일 변경")
    if code is not None and is_limit(res, err):
        reasons = ["한도 도달"]
    else:
        reasons += [f"조작 확인 실패: {m}" for m in manipulation_check(init, row["condition"], model)]
    meta = {"try": try_no, "status": "fail" if reasons else "ok", "reasons": reasons, "exit_code": code,
            "seconds": round(secs, 1), "init": init,
            "result": {k: v for k, v in res.items() if k != "result"},
            "audit": aud, "inputs_changed": changed, "lchome_files": rd.lchome_files,
            "findings_block": bool(re.search(r"```findings", res.get("result") or "")),
            "permission_denials": denials(res)}
    shutil.rmtree(rd.root, ignore_errors=True)
    return Attempt(meta["status"], reasons, code, secs, events, out, err, meta)


def run_checker(row: dict, rid: str) -> dict:
    """(가) 실행마다 점검기 출력을 따로 저장한다 (2차 지표). 동결된 점검기를 그대로 부른다."""
    tmp = RUN_BASE / f"{rid}-checker.json"
    p = subprocess.run(["python3", str(SKILL_SRC / "scripts" / "run_check.py"), str(VARIANT_DIR / row["variant"]),
                        "--data", str(DATA_DIR), "--json", str(tmp)],
                       env={**os.environ, "LEAKCHECK_HOME": str(ROOT)}, capture_output=True)
    out = json.loads(tmp.read_text(encoding="utf-8")) if tmp.exists() else None
    tmp.unlink(missing_ok=True)
    return {"report_id": rid, "variant": row["variant"], "exit_code": p.returncode, "checker": out}


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)


def _fresh(d: Path) -> Path:
    """이어 실행으로 같은 이름이 생겨도 앞 기록을 덮어쓰지 않는다."""
    n, cand = 1, d
    while cand.exists():
        n += 1
        cand = d.with_name(f"{d.name}_{n}")
    return cand


def _gz(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(data, mtime=0))


class LimitReached(Exception):
    """한도에 걸리면 이 행은 시도로 세지 않고 멈춘다. 다음 세션에서 이어 간다."""


def run_row(out: Path, rid: str, row: dict, model: str, users: OsUser, launcher=launch,
            base: Path = RUN_BASE, checker=run_checker) -> dict:
    tries = []
    for k in range(1, MAX_RETRIES + 2):
        a = run_once(rid, row, k, model, users, launcher, base)
        if a.reasons == ["한도 도달"]:
            d = out / "discarded" / rid / f"limit{int(time.time())}"
            _gz(d / "transcript.jsonl.gz", a.raw)
            _gz(d / "stderr.txt.gz", a.stderr)
            _write_json(d / "attempt.json", a.meta)
            raise LimitReached(rid)
        tries.append(a.meta)
        if a.status == "ok":
            break
        d = _fresh(out / "discarded" / rid / f"try{k}")
        _gz(d / "transcript.jsonl.gz", a.raw)
        _gz(d / "stderr.txt.gz", a.stderr)
        _write_json(d / "attempt.json", {**a.meta, "result_text": result_info(a.events).get("result")})
    final = tries[-1]
    meta = {"report_id": rid, "status": "done" if final["status"] == "ok" else "failed",
            "attempts": tries, "retries": len(tries) - 1,
            "retry_reasons": [r for t in tries[:-1] for r in t["reasons"]] if final["status"] == "ok"
            else [r for t in tries for r in t["reasons"]]}
    if final["status"] == "ok":
        _gz(out / "transcripts" / f"{rid}.jsonl.gz", a.raw)
        text = result_info(a.events).get("result") or ""
        _write_json(out / "reports" / f"{rid}.json", {"report_id": rid, "variant": row["variant"], "text": text})
        if row["condition"] == "가":
            _write_json(out / "checker" / f"{rid}.json", checker(row, rid))
    _write_json(out / "meta" / f"{rid}.json", meta)
    return meta


# ---------------------------------------------------------------- 실행 묶음

def pending(out: Path, sched: dict[str, dict]) -> list[str]:
    done = set()
    for rid in sched:
        m = out / "meta" / f"{rid}.json"
        if m.exists() and json.loads(m.read_text(encoding="utf-8"))["status"] == "done":
            done.add(rid)
    return [rid for rid in sorted(sched, key=lambda r: sched[r]["order"]) if rid not in done]


def batches(sched: dict[str, dict]) -> list[int]:
    return sorted({r.get("batch", r["rep"]) for r in sched.values()})


def run_all(out: Path, sched: dict[str, dict], model: str, workers: int = WORKERS, limit: int | None = None,
            users: OsUser | None = None, launcher=launch, base: Path = RUN_BASE, checker=run_checker) -> dict:
    """끝나지 않은 행을 순서대로 돌린다. 한도에 걸리면 새 행을 시작하지 않고 멈춘다.
    돌려주는 값: {"results": [...], "limit_reached": bool}"""
    users = users or OsUser()
    todo = pending(out, sched)[:limit]
    lock = threading.Lock()
    results, stop = [], threading.Event()

    def one(rid: str) -> None:
        if stop.is_set():
            return
        try:
            m = run_row(out, rid, sched[rid], model, users, launcher, base, checker)
        except LimitReached:
            stop.set()
            print(f"{rid} 한도 도달: 새 실행을 시작하지 않는다", flush=True)
            return
        with lock:
            results.append(m)
            print(f"[{len(results)}/{len(todo)}] {rid} {m['status']} 재실행 {m['retries']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    return {"results": results, "limit_reached": stop.is_set()}


def git_commit_push(paths: list[Path], message: str, tries: int = 4) -> bool:
    subprocess.run(["git", "-C", str(ROOT), "add", *map(str, paths)], check=True)
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0:
        return True
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", message], check=True)
    for i in range(tries):
        if subprocess.run(["git", "-C", str(ROOT), "push", "-q", "-u", "origin", "HEAD"]).returncode == 0:
            return True
        time.sleep(2 ** (i + 1))
    return False


COMMIT_TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                  "Claude-Session: https://claude.ai/code/session_01Pac45UtTMCsC9nGygoqhAZ")


def run_batches(out: Path, sched: dict[str, dict], model: str, workers: int, commit: bool = True, **kw) -> bool:
    """묶음 단위로 돌리고, 묶음이 끝날 때마다 커밋·푸시한다. 한도에 걸리면 지금까지를 커밋하고 False."""
    for b in batches(sched):
        sub = {rid: r for rid, r in sched.items() if r.get("batch", r["rep"]) == b}
        if not pending(out, sub):
            continue
        res = run_all(out, sub, model, workers, **kw)
        left = len(pending(out, sub))
        msg = (f"6단계 {out.name}: 묶음 {b} " + ("완료" if left == 0 and not res["limit_reached"]
                                               else f"중단 (남은 {left}회, 한도 도달={res['limit_reached']})"))
        if commit:
            git_commit_push([out], msg + COMMIT_TRAILER)
        print(msg, flush=True)
        if res["limit_reached"]:
            return False
    return True


def status(out: Path) -> dict:
    sched = json.loads((out / "conditions.json").read_text(encoding="utf-8"))
    s = {c: {"planned": 0, "done": 0, "failed": 0, "retries": 0, "retry_reasons": {}, "checker_invoked": 0,
             "findings_block": 0, "seconds": [], "cost_usd": 0.0,
             "permission_denials": 0, "checker_denied": 0, "tokens": {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0}} for c in CONDITIONS}
    for rid, row in sched.items():
        c = s[row["condition"]]
        c["planned"] += 1
        m = out / "meta" / f"{rid}.json"
        if not m.exists():
            continue
        meta = json.loads(m.read_text(encoding="utf-8"))
        c[meta["status"]] += 1
        c["retries"] += meta["retries"]
        for r in meta["retry_reasons"]:
            c["retry_reasons"][r] = c["retry_reasons"].get(r, 0) + 1
        last = meta["attempts"][-1]
        c["checker_invoked"] += last["audit"]["checker_invoked"]
        c["findings_block"] += last["findings_block"]
        c["permission_denials"] += last.get("permission_denials", {}).get("count", 0)
        c["checker_denied"] += last.get("permission_denials", {}).get("checker_denied", 0)
        for t in meta["attempts"]:
            c["seconds"].append(t["seconds"])
            c["cost_usd"] += (t["result"] or {}).get("total_cost_usd") or 0
            for mu in ((t["result"] or {}).get("modelUsage") or {}).values():
                for k, uk in (("input", "inputTokens"), ("cache_creation", "cacheCreationInputTokens"),
                              ("cache_read", "cacheReadInputTokens"), ("output", "outputTokens")):
                    c["tokens"][k] += mu.get(uk) or 0
    for c in s.values():
        secs = c.pop("seconds")
        c["mean_seconds_per_attempt"] = round(sum(secs) / len(secs), 1) if secs else None
        c["cost_usd"] = round(c["cost_usd"], 3)
        n = c["done"] + c["failed"]
        c["mean_tokens_per_run"] = {k: round(v / n) for k, v in c["tokens"].items()} if n else None
    return s


def preflight(base: Path = RUN_BASE) -> list[str]:
    """실행 사용자가 저장소·세션 기록을 읽을 수 없는지 확인한다. 문제 목록을 돌려준다."""
    SHARED.mkdir(parents=True, exist_ok=True)
    SHARED.chmod(0o755)
    shutil.copy2(CA_SRC, SHARED / "ca-bundle.crt")
    (SHARED / "ca-bundle.crt").chmod(0o644)
    base.mkdir(parents=True, exist_ok=True)
    base.chmod(0o711)
    for d in HARDEN_DIRS:
        if d.exists():
            d.chmod(d.stat().st_mode & ~0o007)
    name = "lrpreflight"
    rd = prepare_run(base, "preflight", variants()[0], "나")
    users = OsUser()
    users.create(name, rd)
    probs = []
    try:
        for p in [ROOT / "CLAUDE.md", ROOT / "docs", Path("/root/.claude"), Path("/mnt/user-data"),
                  Path("/tmp/claude-0"), base]:
            if p.exists():
                r = subprocess.run(users.wrap(name, ["ls", str(p)], {"PATH": "/usr/bin:/bin"}), capture_output=True)
                if r.returncode == 0:
                    probs.append(f"실행 사용자가 읽을 수 있음: {p}")
        r = subprocess.run(users.wrap(name, ["ls", str(rd.ws)], {"PATH": "/usr/bin:/bin"}), capture_output=True)
        if r.returncode != 0:
            probs.append("실행 사용자가 자기 실행 폴더를 못 읽음")
    finally:
        users.remove(name, rd)
        shutil.rmtree(rd.root, ignore_errors=True)
    return probs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["schedule", "preflight", "pilot", "main", "status"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(status(a.out or RUNS_OUT), ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "preflight":
        probs = preflight()
        print("\n".join(probs) or "격리 확인 통과")
        return 1 if probs else 0
    if a.cmd == "pilot":
        out, sched = PILOT_OUT, build_schedule(pilot_variants(), 1, PILOT_SEED)
    else:
        out, sched = RUNS_OUT, build_schedule(variants(), REPS, SCHEDULE_SEED)
    sched = ensure_schedule(out, sched)
    if a.cmd == "schedule":
        print(f"{out / 'conditions.json'}: {len(sched)}행")
        return 0
    probs = preflight()
    if probs:
        raise SystemExit("격리 확인 실패:\n" + "\n".join(probs))
    prompt.text()
    finished = run_batches(out, sched, a.model, a.workers, commit=not a.no_commit, limit=a.limit)
    st = status(out)
    print(json.dumps(st, ensure_ascii=False, indent=1))
    if not finished:
        return 3
    return 1 if any(c["failed"] for c in st.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
