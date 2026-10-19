"""Preserve author CPUHead rows and native pooling index mapping."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

import httpx

spec = importlib.util.spec_from_file_location("neohorse_backend", Path(__file__).parents[1] / "decision-eval/neohorse_backend.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NeoHorseAdapterTests(unittest.TestCase):
    def backend(self):
        backend = module.NeoHorseBackend.__new__(module.NeoHorseBackend)
        backend.head = Mock()
        backend.head.prepare.side_effect = [[{"ids": [1, 2]}], [{"ids": [3]}, {"ids": [4]}]]
        backend.head.result.side_effect = lambda rows, hidden: {"answers": hidden}
        backend.url, backend.timeout = "http://localhost/pooling", 300
        backend.health_url = "http://localhost/health"
        backend.session = Mock()
        backend.session.post.return_value.json.return_value = {"data": [
            {"index": 2, "data": "last"}, {"index": 0, "data": "first"}, {"index": 1, "data": "middle"}]}
        return backend

    def test_readiness_requires_native_health_without_inference(self):
        backend = self.backend()
        for status in (503, 500, 200, 503):
            with self.subTest(status=status):
                backend.session.get.return_value.status_code = status
                self.assertEqual(backend.is_ready(), status == 200)
        backend.session.get.assert_called_with("http://localhost/health", timeout=3)
        backend.session.post.assert_not_called()
        backend.head.prepare.assert_not_called()

    def test_readiness_handles_connection_and_timeout_errors(self):
        backend = self.backend()
        for error in (httpx.ConnectError("private dependency"), httpx.ReadTimeout("private dependency")):
            with self.subTest(error=type(error).__name__):
                backend.session.get.side_effect = error
                self.assertFalse(backend.is_ready())
        backend.session.post.assert_not_called()

    def test_batch_maps_rows_back_to_each_original_request(self):
        backend = self.backend()
        results = backend.batch([{"state": "a"}, {"state": "b"}])
        self.assertEqual([row["answers"] for row in results], [["first"], ["middle", "last"]])
        self.assertEqual(backend.session.post.call_args.kwargs["json"]["input"], [[1, 2], [3], [4]])
        self.assertEqual(backend.head.result.call_count, 2)

    def test_duplicate_or_missing_native_indices_fail(self):
        backend = self.backend()
        backend.session.post.return_value.json.return_value = {"data": [
            {"index": 0, "data": []}, {"index": 0, "data": []}, {"index": 2, "data": []}]}
        with self.assertRaises(RuntimeError):
            backend.batch([{}, {}])
        backend.head.result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
