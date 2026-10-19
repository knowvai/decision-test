import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[3] / "docs/research/2026-10-19-evidence-reference-check.py"
spec = importlib.util.spec_from_file_location("evidence_reference", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_controls_balanced_unique():
    jobs = m.controls()
    assert len(jobs) == len({m.key(j) for j in jobs}) == 24
    assert sum(j["gold"] == "relevant" for j in jobs) == 12


def test_selection_independent_of_labels_and_input_order():
    jobs = [{"case_id": str(i), "fragment_id": "p", "gold": "relevant"} for i in range(643)]
    picked = [m.key(j) for j in m.select_reverse(jobs)]
    changed = [{**j, "gold": "irrelevant"} for j in reversed(jobs)]
    assert picked == [m.key(j) for j in m.select_reverse(changed)]
    assert len(set(picked)) == 128


def test_reverse_preserves_names_and_changes_hash():
    q = {"relevance": {"type": "choice", "criteria": {"relevant": "yes", "irrelevant": "no"}}}
    reversed_q = m.reverse(q)
    assert list(q["relevance"]["criteria"]) == ["relevant", "irrelevant"]
    assert list(reversed_q["relevance"]["criteria"]) == ["irrelevant", "relevant"]
    assert m.request_digest({}, q) != m.request_digest({}, reversed_q)


def test_partial_lines_rejected(tmp_path):
    path = tmp_path / "rows"
    path.write_text('{}')
    with pytest.raises(AssertionError):
        m.rows(path)


def test_output_never_overwritten(tmp_path):
    path = tmp_path / "result.json"
    m.save(path, {"a": 1})
    with pytest.raises(FileExistsError):
        m.save(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def answer():
    return {"answers": {"relevance": {"choice": "relevant", "probabilities": {"relevant": .8, "irrelevant": .2}}},
            "usage": {"input_tokens": 42, "output_tokens": 0}}


def test_answer_and_full_input():
    assert m.validate_answer(answer(), 42)[0] == "relevant"
    with pytest.raises(AssertionError):
        m.validate_answer(answer(), 43)


@pytest.mark.parametrize("limit", [None, 3, 4])
def test_encoding_optional_limit_preserves_full_input(limit):
    ids = [11, 12, 13]
    m.validate_encoding("full text", "full text", ids, ids.copy(), limit)
    assert ids == [11, 12, 13]


@pytest.mark.parametrize("limit", [0, -1, True, "3", 2])
def test_encoding_rejects_invalid_or_exceeded_limit(limit):
    with pytest.raises(AssertionError):
        m.validate_encoding("full text", "full text", [11, 12, 13], [11, 12, 13], limit)


@pytest.mark.parametrize("worker_text,worker_ids", [("changed", [11, 12, 13]),
                                                  ("full text", [11, 12]),
                                                  ("full text", [11, 12, 14])])
def test_unset_limit_still_rejects_changed_or_truncated_input(worker_text, worker_ids):
    with pytest.raises(AssertionError):
        m.validate_encoding("full text", worker_text, [11, 12, 13], worker_ids, None)


def test_empty_encoding_rejected():
    with pytest.raises(AssertionError):
        m.validate_encoding("", "", [], [], None)


@pytest.mark.parametrize("probabilities", [{"relevant": float('nan'), "irrelevant": .2},
                                         {"relevant": .2, "irrelevant": .8},
                                         {"relevant": .8, "irrelevant": .3}])
def test_bad_answer_rejected(probabilities):
    raw = answer()
    raw["answers"]["relevance"]["probabilities"] = probabilities
    with pytest.raises(AssertionError):
        m.validate_answer(raw, 42)


def test_pairing_and_counts():
    jobs = m.controls()
    records = [{**j, "prediction": j["gold"], "probabilities": {"relevant": .8}} for j in jobs]
    assert m.confusion(jobs, records) == {"tp": 12, "tn": 12}
    assert m.compare(records, records)["changed"] == 0
    with pytest.raises(AssertionError):
        m.compare(records, records[::-1])
