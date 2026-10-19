"""Offline-only statistical and audit negative cases; no model execution."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("runtime_audit", Path(__file__).resolve().parents[3] /
                                             "docs/research/2026-10-19-d3-runtime-audit.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_compare_by_id_not_record_order():
    rows = {"a": {"prediction": "yes", "probabilities": {"yes": .6, "no": .4}},
            "b": {"prediction": "no", "probabilities": {"yes": .2, "no": .8}}}
    assert a.compare(rows, dict(reversed(list(rows.items()))))["decision_changes"] == 0
    changed = {**rows, "a": {"prediction": "no", "probabilities": {"yes": .4, "no": .6}}}
    assert a.compare(changed, rows)["decision_changes"] == 1
    assert a.compare(changed, rows)["max_probability_delta"] == pytest.approx(.2)
    with pytest.raises(AssertionError):
        a.compare(rows, {"a": rows["a"]})


@pytest.mark.parametrize("event", ["socket.__new__", "socket.connect", "subprocess.Popen", "os.system", "os.posix_spawn"])
def test_offline_denies_side_effects(event):
    with pytest.raises(RuntimeError):
        a.offline(event, ())


def test_repeated_inputs_not_counted_as_new_quality_samples():
    jobs = [dict(case_id="x", fragment_id="1", gold="relevant"),
            dict(case_id="x", fragment_id="2", gold="irrelevant")]
    rows = [dict(case_id="x", fragment_id="1", prediction="relevant"),
            dict(case_id="x", fragment_id="2", prediction="relevant")] * 2
    value = a.quality(rows, jobs, "evidence")
    assert value["unique_inputs"] == 2 and value["accuracy"] == .5
    assert value["precision"] == .5 and value["recall"] == 1 and value["f1"] == pytest.approx(2/3)


def test_cpu_delta_requires_same_quota_and_monotonic_counters():
    before = dict(quota="800000 100000", stats="usage_usec 1000\nnr_periods 1\nnr_throttled 0\n")
    after = dict(quota=before["quota"], stats="usage_usec 2000\nnr_periods 3\nnr_throttled 1\n")
    assert a.cpu_delta(dict(cpu_before=before, cpu_after=after)) == dict(cpu_seconds=.001, periods=2, throttled_periods=1)
    with pytest.raises(AssertionError):
        a.cpu_delta(dict(cpu_before=after, cpu_after=before))
