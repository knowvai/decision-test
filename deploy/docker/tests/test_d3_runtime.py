"""Parameter isolation, no-retry admission and native automatic-budget regression."""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("d3_runtime", ROOT / "docs/research/2026-10-19-d3-runtime-run.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def jobs(n):
    return [dict(case_id=str(i), fragment_id="f", input_sha256=str(i), chars=i) for i in range(n)]


def test_matrix_count_and_one_factor():
    rows = list(r.blocks(jobs(128)))
    assert len(rows) == 12
    assert 2 * 9 * 2 * sum(len(x[1]) for x in rows) == 41472
    assert 2 * 9 * 2 * len(rows) == 432
    assert r.profile_order("d3-flash")[1:-1] == list(reversed(r.profile_order("d3-mini")[1:-1]))
    for name, profile in r.PROFILES.items():
        if name not in ("baseline", "baseline-end", "auto"):
            assert sum(profile[k] != r.BASE[k] for k in r.BASE) == 1
    for name, rows, n, phase, repeat in rows:
        assert len({j["case_id"] for j in rows}) == len(rows)


def test_fail_closed_no_new_http_after_failure():
    sent = []
    async def infer(job):
        sent.append(job["case_id"])
        raise ValueError("synthetic failure")
    rows, _ = asyncio.run(r.fail_closed_measure(jobs(20), infer, 4, 1))
    assert len(sent) == 1
    assert rows[0]["status"] == "failed"
    assert all(row["status"] == "not_attempted" and row["calls"] == 0 for row in rows[1:])


def test_timeout_closes_admission():
    sent = []
    async def infer(job):
        sent.append(job["case_id"])
        await asyncio.sleep(1)
    rows, _ = asyncio.run(r.fail_closed_measure(jobs(20), infer, 4, .01))
    assert len(sent) == 4
    assert all(row["status"] == "failed" for row in rows[:4])
    assert all(row["status"] == "not_attempted" for row in rows[4:])


@pytest.mark.parametrize("auto", [False, True])
def test_native_flags_and_defaults(auto):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "models/d3-mini").mkdir(parents=True)
        args = SimpleNamespace(root=root, model="d3-mini", env_file=root / ".env", native_port=30888,
                               public_port=30889, enforce_eager=False, worker_auto_budget=auto, **r.BASE)
        def docker(*parts):
            return "" if parts[-1] == "{{.Config.User}}" else "sha256:" + "a" * 64
        with patch.object(r.launcher, "docker", side_effect=docker), patch.object(r.launcher, "remove_owned"), \
             patch.object(r.launcher, "read_key", return_value="A1" * 16), \
             patch.object(r.launcher.socket, "socket"), patch.object(r.launcher.subprocess, "check_output", return_value=""):
            r.launcher.launch(args)
        record = json.loads((root / "acceptance-20261015/evidence/d3-mini.deployment.json").read_text())
        assert ("--max-num-seqs" in record["command"]) is not auto
        assert ("--max-num-batched-tokens" in record["command"]) is not auto
        assert record["scheduling"]["max_sequences"] == (None if auto else 32)
        assert record["scheduling"]["max_batched_tokens"] == (None if auto else 32768)


@pytest.mark.parametrize("model,auto", [("clef-flash", True), ("d3-mini", "yes"), ("d3-mini", 1)])
def test_invalid_auto_no_service_side_effects(model, auto):
    with tempfile.TemporaryDirectory() as tmp, patch.object(r.launcher, "remove_owned") as remove:
        with pytest.raises(ValueError):
            r.launcher.launch(SimpleNamespace(root=Path(tmp), model=model, worker_auto_budget=auto))
        remove.assert_not_called()
