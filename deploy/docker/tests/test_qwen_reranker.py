"""Qwen evaluation boundaries without downloading or loading a GPU model."""

import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).parents[1] / "decision-eval"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


server = load("qwen_eval_server", ROOT / "server.py")
adapter = load("qwen_eval_adapter", ROOT / "qwen_reranker.py")
KEY = "test-credential-not-for-deployment-123456789"
SETTINGS = {"DECISION_EVAL_API_KEY": KEY, "MODEL_NAME": "fixture", "MODEL_BACKEND": "qwen_reranker"}
BODY = {"model": "fixture", "query": " query ", "documents": [" first ", "second"], "batch_size": 4}
HEADERS = {"Authorization": "Bearer " + KEY}


class QwenServiceTests(unittest.TestCase):
    def client(self, operation=None):
        operation = operation or (lambda body: {"model": body["model"], "results": [
            {"index": i, "relevance_score": 0.25 * i} for i in range(len(body["documents"]))]})
        return TestClient(server.create_app(settings=SETTINGS, rerank=operation))

    def test_requires_key_on_all_paths(self):
        with self.client() as client:
            for path in ["/v1/rerank", "/v1/systemone", "/v1/decisions", "/health"]:
                request = client.get if path == "/health" else client.post
                self.assertEqual(request(path).status_code, 401)
                self.assertEqual(request(path, headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_preserves_all_documents_and_indices_without_top_k(self):
        operation = Mock(return_value={"results": [{"index": 0, "relevance_score": 0.1},
                                                   {"index": 1, "relevance_score": 0.9}]})
        with self.client(operation) as client:
            response = client.post("/v1/rerank", json=BODY, headers=HEADERS)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["results"], operation.return_value["results"])
            operation.assert_called_once_with(BODY)
            self.assertGreaterEqual(response.json()["latency_ms"], 0)

    def test_rejects_invalid_shapes_options_and_model_switching(self):
        invalid = [{**BODY, "model": "other"}, {**BODY, "query": " "},
                   {**BODY, "query": 10}, {**BODY, "documents": []},
                   {**BODY, "documents": [" "]}, {**BODY, "documents": [12]},
                   {**BODY, "documents": ["doc"] * 129}, {**BODY, "batch_size": True},
                   {**BODY, "batch_size": 1.0}, {**BODY, "batch_size": "4"},
                   {**BODY, "batch_size": 2}, {**BODY, "top_n": 1},
                   {**BODY, "url": "http://example.invalid"}]
        operation = Mock()
        with self.client(operation) as client:
            for body in invalid:
                self.assertEqual(client.post("/v1/rerank", json=body, headers=HEADERS).status_code, 422)
        operation.assert_not_called()

    def test_decision_and_rerank_backends_do_not_substitute_for_each_other(self):
        decision = {"model": "fixture", "state": "text", "questions": {"q": {"type": "noul"}}}
        with self.client() as client:
            self.assertEqual(client.post("/v1/systemone", json=decision, headers=HEADERS).status_code, 422)
        with TestClient(server.create_app(predict=lambda body: {}, settings=SETTINGS)) as client:
            self.assertEqual(client.post("/v1/rerank", json=BODY, headers=HEADERS).status_code, 422)

    def test_busy_inference_is_bounded_and_gate_recovers(self):
        entered, release = threading.Event(), threading.Event()

        def operation(body):
            entered.set()
            release.wait(5)
            return {"results": []}

        with self.client(operation) as client:
            replies = []
            thread = threading.Thread(target=lambda: replies.append(client.post("/v1/rerank", json=BODY, headers=HEADERS)))
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(client.post("/v1/rerank", json=BODY, headers=HEADERS).status_code, 429)
            finally:
                release.set()
                thread.join(5)
            self.assertEqual(replies[0].status_code, 200)
            self.assertEqual(client.post("/v1/rerank", json=BODY, headers=HEADERS).status_code, 200)

    def test_errors_are_redacted_and_oom_is_not_retried(self):
        for error, status in [(ValueError("private input"), 422),
                              (RuntimeError("private runtime"), 500),
                              (RuntimeError("CUDA out of memory private detail"), 503)]:
            operation = Mock(side_effect=error)
            torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))
            with patch.dict(sys.modules, {"torch": torch}), self.client(operation) as client:
                response = client.post("/v1/rerank", json=BODY, headers=HEADERS)
                self.assertEqual(response.status_code, status)
                self.assertNotIn("private", response.text)
                operation.assert_called_once()
                self.assertEqual(client.get("/health", headers=HEADERS).status_code, 200)


class QwenEncodingTests(unittest.TestCase):
    def tokenizer(self, lengths):
        tokenizer = Mock(return_value={"input_ids": [list(range(n)) for n in lengths]})
        tokenizer.encode.side_effect = [[91, 92], [93]]
        return tokenizer

    def test_official_template_and_raw_inputs_are_preserved(self):
        tokenizer = self.tokenizer([2, 3])
        records = adapter.encode_records(tokenizer, BODY, 10)
        self.assertEqual(records, [{"input_ids": [91, 92, 0, 1, 93]},
                                   {"input_ids": [91, 92, 0, 1, 2, 93]}])
        expected = [f'<Instruct>: {adapter.DEFAULT_INSTRUCTION}\n<Query>: {BODY["query"]}\n<Document>: {d}'
                    for d in BODY["documents"]]
        tokenizer.assert_called_once_with(expected, padding=False, truncation=False,
                                          add_special_tokens=False, return_attention_mask=False)
        self.assertEqual(tokenizer.encode.call_args_list[0].args[0], adapter.PREFIX)
        self.assertEqual(tokenizer.encode.call_args_list[1].args[0], adapter.SUFFIX)

    def test_late_overlength_document_fails_entire_pool(self):
        with self.assertRaises(ValueError):
            adapter.encode_records(self.tokenizer([2, 8]), BODY, 10)
        self.assertEqual(len(adapter.encode_records(self.tokenizer([2, 7]), BODY, 10)), 2)

    def test_incomplete_tokenizer_output_is_not_a_successful_pool(self):
        with self.assertRaises(RuntimeError):
            adapter.encode_records(self.tokenizer([2]), BODY, 10)


if __name__ == "__main__":
    unittest.main()
