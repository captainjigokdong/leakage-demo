import csv
import json
import subprocess
import sys
from pathlib import Path

from tests.design_mutations import FIXTURES, inject

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skill_src" / "leakage-check"
RUN = SKILL / "scripts" / "run_check.py"


def test_skill_not_in_autoload_location():
    assert not (ROOT / ".claude" / "skills").exists()
    assert (SKILL / "SKILL.md").exists()


def test_skill_md_has_frontmatter_and_no_public_case_examples():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: leakage-check\n")
    with open(ROOT / "designs" / "error_catalog_public.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["id"] not in text
            assert r["name"] not in text
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"):
        assert q in text
    assert "run_check.py" in text


def _run(*args):
    return subprocess.run([sys.executable, str(RUN), *map(str, args)], capture_output=True, text=True, cwd=ROOT)


def test_cli_exit_codes(tmp_path):
    ok = _run(FIXTURES / "clean_fixed_readmission.json")
    assert ok.returncode == 0, ok.stderr
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(inject("E08", "고정"), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out.json"
    blocked = _run(bad, "--json", out)
    assert blocked.returncode == 1
    assert json.loads(out.read_text(encoding="utf-8"))["blocked"] is True
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"design_id": "x"}), encoding="utf-8")
    err = _run(broken)
    assert err.returncode == 2 and "점검 불가" in err.stderr
