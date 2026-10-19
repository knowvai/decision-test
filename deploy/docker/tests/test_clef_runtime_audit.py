"""Pure offline Clef audit regression tests; no inference."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("clef_audit", ROOT / "docs/research/2026-10-19-clef-runtime-audit.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_nested_settings():
    assert a.values({"server_args": {"schedule": {"max_running_requests": 32}}}, "max_running_requests") == [32]
    assert a.values({"internal_states": [{"max_req_len": 8191}]}, "max_req_len") == [8191]
    assert a.values({}, "missing") == []


def test_concurrency_recomputed():
    rows = [{"started_seconds": 0., "ended_seconds": .1, "elapsed_ms": 100.},
            {"started_seconds": .02, "ended_seconds": .12, "elapsed_ms": 100.}]
    a.check_intervals(rows, 2, .13, 2)
    with pytest.raises(AssertionError):
        a.check_intervals(rows, 1, .13, 1)
    with pytest.raises(AssertionError):
        a.check_intervals(rows, 2, .11, 2)


def test_pairing_not_row_order():
    first = {("a", 1): {"prediction": "a", "probabilities": {"a": .7, "b": .3}},
             ("b", 1): {"prediction": "b", "probabilities": {"a": .3, "b": .7}}}
    second = dict(reversed(list(first.items())))
    assert a.helper.compare(first, second)["decision_changes"] == 0
    with pytest.raises(AssertionError):
        a.helper.compare(first, {("a", 1): first["a", 1]})


@pytest.mark.parametrize("event", ["socket.connect", "socket.__new__", "subprocess.Popen", "os.system", "os.posix_spawn"])
def test_offline_guard(event):
    with pytest.raises(RuntimeError):
        a.helper.offline(event, ())


def test_quality_keeps_independent_input_count():
    jobs = [{"case_id": "a", "fragment_id": "f", "gold": "relevant"},
            {"case_id": "b", "fragment_id": "f", "gold": "irrelevant"}]
    rows = [{"case_id": "a", "fragment_id": "f", "prediction": "relevant"},
            {"case_id": "b", "fragment_id": "f", "prediction": "relevant"}] * 2
    result = a.helper.quality(rows, jobs, "evidence")
    assert result["unique_inputs"] == 2 and result["accuracy"] == .5
    assert result["precision"] == .5 and result["recall"] == 1


def test_metrics_counter_type_not_suffix():
    before = "# TYPE running_count gauge\nrunning_count 8\n# TYPE calls_total counter\ncalls_total 10\n"
    after = "# TYPE running_count gauge\nrunning_count 0\n# TYPE calls_total counter\ncalls_total 12\n"
    assert a.metrics_delta({"metrics_before": before, "metrics_after": after}) == {"calls_total": 2}


def test_changed_input_is_not_repeated_event_count():
    first = {("a", "f", n): {"prediction": "yes", "probabilities": {"yes": .51, "no": .49}} for n in (1, 2)}
    second = {("a", "f", n): {"prediction": "no", "probabilities": {"yes": .49, "no": .51}} for n in (1, 2)}
    result = a.compare(first, second)
    assert result["decision_changes"] == 2 and result["unique_inputs_changed"] == 1
