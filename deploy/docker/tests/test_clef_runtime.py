"""Clef adapter, isolation and frozen experimental matrix regressions."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("clef_runtime", ROOT / "docs/research/2026-10-19-clef-runtime-run.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_matrix():
    jobs = [dict(case_id=str(i), fragment_id="f", chars=i) for i in range(128)]
    blocks = list(c.blocks(jobs))
    assert len(blocks) * 2 * len(c.PROFILES) == c.BLOCKS
    assert sum(len(b[1]) for b in blocks) * 2 * len(c.PROFILES) == c.CALLS
    assert c.PROFILES["baseline"] == c.PROFILES["baseline-end"]
    assert len({j["case_id"] for j in blocks[0][1]}) == 32
    assert blocks[4][2] == blocks[-1][2] == 1
    for name, profile in c.PROFILES.items():
        if name not in ("baseline", "wide", "baseline-end"):
            assert sum(profile[k] != c.PROFILES["baseline"][k] for k in profile) == 1


@pytest.mark.parametrize("family", ["worker", "sglang"])
def test_native_wire_and_decode(family):
    job = c.t.make_job("private-id", "f", {"query": "public fixture"}, gold="hidden")
    question = {"q": {"type": "choice", "instructions": "choose", "criteria": {"yes": "yes", "no": "no"}}}
    session = c.p.Session(None, None, "redacted", "test-model", native_family=family,
                          expected_tokens={("private-id", "f"): 123})
    path, body = session.request(job, question)
    assert path == "/v1/systemone" and body["state"] == job["state"]
    assert "private-id" not in json.dumps(body) and "hidden" not in json.dumps(body)
    assert ("options" in body) == (family == "worker")
    assert body["questions"]["q"].get("require_full_input", False) == (family == "worker")
    response = {"model": "test-model", "answers": {"q": {"type": "choice", "choice": "yes",
                "probabilities": {"yes": .6, "no": .4}}}, "usage": {"input_tokens": 123, "output_tokens": 0}}
    if family == "worker":
        response["meta"] = {"device": "cuda:0"}
        _, legacy = c.t.wire("test-model", job, question)
        legacy["options"] = {"return_meta": True}
        assert body == legacy
    assert session.decode(body, response, job)[0] == "yes"
    response["usage"]["input_tokens"] -= 1
    with pytest.raises(AssertionError):
        session.decode(body, response, job)


def test_settings_redaction():
    value = {"api_key": "secret", "server_args": {"dtype": "bfloat16", "api_key": "secret",
             "nested": {"max_running_requests": 32}}, "internal_states": [{"max_req_len": 8191, "prompt": "private"}]}
    result = c.safe_settings(value)
    assert "secret" not in json.dumps(result) and "private" not in json.dumps(result)
    assert result["server_args"]["nested"]["max_running_requests"] == 32


@pytest.mark.parametrize("metrics", [False, True])
def test_metrics_opt_in_preserves_defaults(metrics):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "models/clef-flash").mkdir(parents=True)
        args = SimpleNamespace(root=root, model="clef-flash", env_file=root / ".env", native_port=30888,
                               public_port=30889, seqs=64, tokens=65536, enforce_eager=False, sglang_metrics=metrics)
        with patch.object(c.launcher, "docker", return_value="sha256:" + "a" * 64), \
             patch.object(c.launcher, "remove_owned"), patch.object(c.launcher, "read_key", return_value="A1" * 16), \
             patch.object(c.launcher.socket, "socket"), patch.object(c.launcher.subprocess, "check_output", return_value=""):
            c.launcher.launch(args)
        cmd = json.loads((root / "acceptance-20261015/evidence/clef-flash.deployment.json").read_text())["command"]
        assert ("--enable-metrics" in cmd) == metrics
        assert cmd[cmd.index("--max-running-requests") + 1] == "64"
        assert cmd[cmd.index("--max-prefill-tokens") + 1] == "65536"
        assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.85"


@pytest.mark.parametrize("model,metrics", [("d3-mini", True), ("clef-flash", 1), ("clef-flash", "true")])
def test_metrics_rejects_other_families_before_mutation(model, metrics):
    with tempfile.TemporaryDirectory() as tmp, patch.object(c.launcher, "remove_owned") as remove:
        with pytest.raises(ValueError):
            c.launcher.launch(SimpleNamespace(root=Path(tmp), model=model, sglang_metrics=metrics))
        remove.assert_not_called()
