"""Native-independent pool boundaries without importing GPU dependencies."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


path = Path(__file__).parents[1] / "decision-eval/decision_reranker.py"
spec = importlib.util.spec_from_file_location("decision_reranker_under_test", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
QUESTION = {"type": "score", "criteria": ["none", "topic", "partial", "direct"]}
CHOICE_QUESTION = {"type": "choice", "criteria": {"relevant": "Relevant", "irrelevant": "Irrelevant"}}
BODY = {"model": "test", "query": "query", "documents": ["one", "two", "three", "four", "five"],
        "batch_size": 4, "questions": {"relevance": QUESTION}}


@pytest.mark.parametrize("probabilities", [
    {"0": 0.2, "1": 0.3}, {"0": 0, "1": 0, "2": 0, "3": 0},
    {"0": 0, "1": 0, "2": 0, "3": float("nan")},
    {"0": -1, "1": 0, "2": 0, "3": 2}, {"0": 0, "1": 0, "2": 0, "3": True},
])
def test_invalid_native_distribution_fails(probabilities):
    with pytest.raises(RuntimeError):
        adapter.score_answer({"type": "score", "probabilities": probabilities}, QUESTION)


def test_expected_grade_mapping_not_confidence_gate():
    answer = {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}, "confidence": 0.01}
    assert adapter.score_answer(answer, QUESTION) == pytest.approx(2 / 3)


def test_choice_mapping_uses_relevant_probability():
    answer = {"type": "choice", "probabilities": {"relevant": 0.73, "irrelevant": 0.27}}
    assert adapter.score_answer(answer, CHOICE_QUESTION) == pytest.approx(0.73)


class ForwardTarget:
    def register_forward_pre_hook(self, callback, with_kwargs):
        assert with_kwargs
        self.callback = callback
        return SimpleNamespace(remove=lambda: setattr(self, "callback", None))

    def parameters(self):
        return iter([SimpleNamespace(dtype="torch.bfloat16")])


def fixture_runtime(reject=None, incomplete=False):
    target = ForwardTarget()
    runtime = SimpleNamespace(backbone=target)

    def prepare(state, questions):
        assert set(state) == {"query", "passage"} and state["query"] == BODY["query"]
        sequence = list(range(len(state["passage"])))
        return SimpleNamespace(errors={"relevance": "overflow"} if state["passage"] == reject else {},
                               runnable=["relevance"], sequences={"relevance": sequence},
                               questions={"relevance": SimpleNamespace(keys=["0", "1", "2", "3"])})

    def probabilities(sequences, counts):
        assert counts == [4] * len(sequences)
        tensor = SimpleNamespace(device=SimpleNamespace(type="cuda", __str__=lambda: "cuda:0"),
                                 shape=(len(sequences), max(map(len, sequences))))
        target.callback(target, (), {"input_ids": tensor})
        return [[0, 0, 0, 1] for _ in sequences[:-1] if incomplete] if incomplete else [[0, 0, 0, 1] for _ in sequences]

    runtime.prepare, runtime.probabilities = prepare, probabilities
    return runtime


def fake_torch():
    return SimpleNamespace(inference_mode=nullcontext, is_autocast_enabled=lambda device: False,
                           cuda=SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                                max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200))


def test_entire_pool_is_validated_before_any_forward():
    runtime = fixture_runtime(reject="five")
    runtime.probabilities = Mock()
    with pytest.raises(ValueError):
        adapter.DecisionReranker("d3", runtime, 8192).prepare(BODY)
    runtime.probabilities.assert_not_called()


def test_real_forward_hook_counts_and_full_independent_scores():
    runtime = fixture_runtime()
    official = SimpleNamespace(product_answer=lambda q, p: {"type": "score", "probabilities": dict(zip(q.keys, p))})
    with patch.dict(sys.modules, {"torch": fake_torch(), "d3_runtime": official}):
        result = adapter.DecisionReranker("d3", runtime, 8192)(BODY)
    assert [r["index"] for r in result["results"]] == list(range(5))
    assert [r["relevance_score"] for r in result["results"]] == [1] * 5
    assert result["execution"]["batch_lengths"] == [4, 1]
    assert result["execution"]["forward_calls"] == 2
    assert len(result["execution"]["observed_forwards"]) == 2
    assert runtime.backbone.callback is None


def test_incomplete_batch_is_not_partial_success_and_hook_is_removed():
    runtime = fixture_runtime(incomplete=True)
    official = SimpleNamespace(product_answer=lambda q, p: {"type": "score", "probabilities": dict(zip(q.keys, p))})
    with patch.dict(sys.modules, {"torch": fake_torch(), "d3_runtime": official}), pytest.raises(RuntimeError):
        adapter.DecisionReranker("d3", runtime, 8192)(BODY)
    assert runtime.backbone.callback is None


def test_unverified_intern_batch_is_explicitly_rejected():
    with pytest.raises(ValueError, match="not validated"):
        adapter.DecisionReranker("intern", None, 8192).prepare(BODY)
