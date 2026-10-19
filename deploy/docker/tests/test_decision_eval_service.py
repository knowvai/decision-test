"""Security and transport checks without importing or loading GPU models."""

import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


spec = importlib.util.spec_from_file_location("decision_eval_server", Path(__file__).parents[1] / "decision-eval" / "server.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
KEY = "test-credential-not-for-deployment-123456789"
SETTINGS = {"DECISION_EVAL_API_KEY": KEY, "MODEL_NAME": "fixture", "MODEL_BACKEND": "fixture"}
BODY = {"model": "fixture", "state": "test data", "questions": {"route": {"type": "choice", "criteria": {"a": "first", "b": "second"}}}}


class DecisionServiceTests(unittest.TestCase):
    def client(self, predict=None, settings=None):
        return TestClient(module.create_app(predict or (lambda body: {"model": "fixture", "answers": {"route": {"type": "choice", "choice": "a", "probabilities": {"a": 0.75, "b": 0.25}}}, "usage": {"input_tokens": 10}}), SETTINGS if settings is None else settings))

    def test_all_paths_require_credentials(self):
        with self.client() as client:
            for path in ["/v1/systemone", "/v1/decisions"]:
                self.assertEqual(client.post(path, json=BODY).status_code, 401)
                self.assertEqual(client.post(path, json=BODY, headers={"Authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(client.get("/health").status_code, 401)

    def test_preserves_native_answer_and_usage(self):
        with self.client() as client:
            for path in ["/v1/systemone", "/v1/decisions"]:
                response = client.post(path, json=BODY, headers={"Authorization": "Bearer " + KEY})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["answers"]["route"]["probabilities"], {"a": 0.75, "b": 0.25})
                self.assertEqual(response.json()["usage"], {"input_tokens": 10})
                self.assertGreaterEqual(response.json()["latency_ms"], 0)

    def test_health_waits_for_backend_dependency_and_remains_authenticated(self):
        class Backend:
            ready = False
            checks = 0

            def is_ready(self):
                self.checks += 1
                return self.ready

            def __call__(self, body):
                raise AssertionError("Health must not run inference")

        backend = Backend()
        with self.client(backend) as client:
            self.assertEqual(client.get("/health").status_code, 401)
            self.assertEqual(backend.checks, 0)
            headers = {"Authorization": "Bearer " + KEY}
            response = client.get("/health", headers=headers)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "Model dependency is not ready"})
            backend.ready = True
            self.assertEqual(client.get("/health", headers=headers).json(),
                             {"status": "ok", "model": "fixture", "backend": "fixture"})
            backend.ready = False
            self.assertEqual(client.get("/health", headers=headers).status_code, 503)

    def test_health_without_separate_dependency_keeps_existing_payload(self):
        with self.client() as client:
            response = client.get("/health", headers={"Authorization": "Bearer " + KEY})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok", "model": "fixture", "backend": "fixture"})

    def test_rejects_model_switching_media_and_bad_questions(self):
        invalid = [
            {**BODY, "model": "another"},
            {**BODY, "images": ["file:///secret"]},
            {**BODY, "state": 42},
            {**BODY, "questions": {}},
            {**BODY, "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
            {**BODY, "questions": {"q": {"type": "score", "criteria": ["low", 10]}}},
            {**BODY, "questions": {"q": {"type": "noul", "url": "https://example.test"}}},
        ]
        with self.client() as client:
            for body in invalid:
                self.assertEqual(client.post("/v1/systemone", json=body, headers={"Authorization": "Bearer " + KEY}).status_code, 422)

    def test_startup_without_key_fails_closed(self):
        for key in ["", "short"]:
            with self.assertRaises(RuntimeError):
                with self.client(settings={**SETTINGS, "DECISION_EVAL_API_KEY": key}):
                    pass

    def test_clef_rejects_truncation_before_inference(self):
        encode = Mock(return_value=SimpleNamespace(input_ids=list(range(11))))
        infer = Mock(return_value={"answers": {}})
        processor = SimpleNamespace(tokenizer=object())
        official = SimpleNamespace(encode_record=encode, systemone=infer)
        with patch.dict(sys.modules, {"joint_schema_model": official}):
            with self.assertRaises(ValueError):
                module.predict_clef(None, processor, BODY, 10)
            infer.assert_not_called()
            encode.assert_called_with(processor.tokenizer, BODY, max_length=11, processor=processor)
            encode.return_value = SimpleNamespace(input_ids=list(range(10)))
            self.assertEqual(module.predict_clef(None, processor, BODY, 10), {"answers": {}})
            infer.assert_called_once_with(None, processor, BODY, max_length=10)

    def test_runtime_failure_is_redacted_and_gate_recovers(self):
        calls = []

        def predict(body):
            calls.append(body)
            if len(calls) == 1:
                raise RuntimeError("internal details must not reach the caller")
            return {"answers": {}}

        with self.client(predict) as client:
            failed = client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer " + KEY})
            self.assertEqual(failed.status_code, 500)
            self.assertNotIn("internal details", failed.text)
            self.assertEqual(client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer " + KEY}).status_code, 200)

    def test_cpu_retry_is_forbidden_for_decisions_and_reranking(self):
        attempted_retry = Mock()

        def gpu_operation(body):
            warnings.warn("Warning: GPU memory exceeded during inference. Retrying this request on CPU...", RuntimeWarning)
            attempted_retry()
            return {"answers": {}}

        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))
        pool = {"model": "fixture", "query": "question", "documents": ["passage"], "batch_size": 1}
        with patch.dict(sys.modules, {"torch": torch}):
            with TestClient(module.create_app(predict=gpu_operation, settings=SETTINGS)) as client:
                client.app.state.rerank = gpu_operation
                for path, body in [("/v1/systemone", BODY), ("/v1/rerank", pool)]:
                    response = client.post(path, json=body, headers={"Authorization": "Bearer " + KEY})
                    self.assertEqual(response.status_code, 503)
                    self.assertNotIn("Retrying", response.text)
                client.app.state.predict = lambda body: {"answers": {}}
                self.assertEqual(client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer " + KEY}).status_code, 200)
        attempted_retry.assert_not_called()

    def test_laya_rejects_state_and_option_truncation_before_inference(self):
        tok = SimpleNamespace(mask_token="[MASK]")
        agent = SimpleNamespace(tok=tok, _to_internal=lambda q: {"t": "choice", "ins": "instruction"}, predict=Mock(return_value={"answers": {}}))
        common = SimpleNamespace(
            encode_text=Mock(return_value={"input_ids": [1, 2]}),
            render_options=Mock(return_value=["one", "two"]),
            build_head=Mock(return_value=(list(range(11)), [5, 8], {})),
            build_sequence=Mock(return_value=([], [5, 8], {"truncated": True})),
        )
        with patch.dict(sys.modules, {"laya.common": common}):
            with self.assertRaises(ValueError):
                module.predict_laya(agent, BODY, 20)
            agent.predict.assert_not_called()
            common.build_sequence.return_value = ([], [5, 8], {"truncated": False})
            self.assertEqual(module.predict_laya(agent, BODY, 20), {"answers": {}})
            agent.predict.assert_called_once_with(BODY["state"], BODY["questions"], max_len=20, head_max_len=20)
            agent.predict.reset_mock()
            common.encode_text.return_value = {"input_ids": list(range(49))}
            with self.assertRaises(ValueError):
                module.predict_laya(agent, BODY, 200)
            agent.predict.assert_not_called()

    def test_busy_request_returns_429_and_gate_recovers(self):
        entered, release = threading.Event(), threading.Event()

        def predict(body):
            entered.set()
            release.wait(timeout=5)
            return {"answers": {}}

        with self.client(predict) as client:
            first_results = []
            thread = threading.Thread(target=lambda: first_results.append(client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer " + KEY})))
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=2))
                self.assertEqual(client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer " + KEY}).status_code, 429)
            finally:
                release.set()
                thread.join(timeout=5)
            self.assertEqual(first_results[0].status_code, 200)
            self.assertEqual(client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer " + KEY}).status_code, 200)

    def test_configured_inflight_limit_allows_bounded_parallel_requests(self):
        entered, release = [], threading.Event()
        entered_lock = threading.Lock()
        both_entered = threading.Event()

        def predict(body):
            with entered_lock:
                entered.append(body)
                if len(entered) == 2:
                    both_entered.set()
            release.wait(timeout=5)
            return {"answers": {}}

        settings = {**SETTINGS, "MAX_INFLIGHT_REQUESTS": "2"}
        with self.client(predict, settings=settings) as client:
            responses = []
            threads = [threading.Thread(target=lambda: responses.append(
                client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer " + KEY})))
                       for _ in range(2)]
            for thread in threads:
                thread.start()
            try:
                self.assertTrue(both_entered.wait(timeout=2))
                self.assertEqual(
                    client.post("/v1/systemone", json=BODY,
                                headers={"Authorization": "Bearer " + KEY}).status_code,
                    429,
                )
            finally:
                release.set()
                for thread in threads:
                    thread.join(timeout=5)
            self.assertEqual([response.status_code for response in responses], [200, 200])

    def test_single_request_queue_accepts_full_pool_without_claiming_batch(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def predict(body):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            result = {"answers": {}, "state_id": body["state"]}
            with lock:
                active -= 1
            return result

        settings = {**SETTINGS, "SYSTEMONE_QUEUE_SINGLE_REQUESTS": "1", "SYSTEMONE_QUEUE_SIZE": "64"}
        with self.client(predict, settings) as client, ThreadPoolExecutor(max_workers=49) as pool:
            results = list(pool.map(lambda i: client.post(
                "/v1/systemone", json={**BODY, "state": str(i)},
                headers={"Authorization": "Bearer " + KEY}), range(49)))
            self.assertTrue(all(result.status_code == 200 for result in results))
            self.assertEqual([result.json()["state_id"] for result in results], list(map(str, range(49))))
            self.assertEqual(peak, 1)
            for result in results:
                execution = result.json()["execution"]["systemone_batch"]
                self.assertFalse(execution["cross_request_batching"])
                self.assertEqual(execution["native_batch_calls"], 0)

    def test_queue_capacity_deadline_and_shutdown_are_bounded(self):
        worker = module.SystemOneBatcher(lambda bodies: bodies, 1, 0, 1,
                                        native_batching=False, timeout_seconds=0.05)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(worker.submit, BODY)
            for _ in range(100):
                if worker.queue.qsize():
                    break
                threading.Event().wait(0.001)
            with self.assertRaises(module.BatchQueueFull):
                worker.submit(BODY)
            with self.assertRaises(TimeoutError):
                first.result(timeout=2)
        self.assertTrue(worker.queue.queue[0].expired)
        calls = Mock(return_value=[])
        worker.operation = calls
        worker.start()
        worker.close()
        calls.assert_not_called()
        with self.assertRaises(RuntimeError):
            worker.submit(BODY)

    def test_native_batch_preserves_mapping_and_reports_actual_batch(self):
        worker = module.SystemOneBatcher(
            lambda bodies: [{"id": body["state"]} for body in bodies], 4, 30, 64)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker.submit, {**BODY, "state": str(i)}) for i in range(4)]
            for _ in range(100):
                if worker.queue.qsize() == 4:
                    break
                threading.Event().wait(0.001)
            worker.start()
            try:
                results = [future.result(timeout=2) for future in futures]
            finally:
                worker.close()
        self.assertEqual([result["id"] for result in results], ["0", "1", "2", "3"])
        self.assertTrue(all(result["execution"]["systemone_batch"]["batch_size"] == 4 for result in results))


if __name__ == "__main__":
    unittest.main()
