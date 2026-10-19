"""Offline audit guards do not import or call the inference runner."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("d3_audit", ROOT / "docs/research/2026-10-19-d3-tuning-audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.config = {"worker_threads": 2, "seqs": 32, "tokens": 32768, "batch_window_ms": 2}
        self.deployment = {"model": "d3-mini", "served_model": "d3-mini", "family": "worker",
            "tp": 1, "cpu_budget": 8, "host_memory_limit_gib": 96, "image_id": "frozen-image",
            "env": {"OMP_NUM_THREADS": "2"},
            "command": ["vllm-srun", "serve", "/model", "--device", "cuda:0", "--dtype", "auto",
                        "--served-model-name", "d3-mini", "--threads", "2", "--max-num-seqs", "32",
                        "--max-num-batched-tokens", "32768", "--batch-window-ms", "2",
                        "--result-cache-entries", "0", "--load-attempts", "1",
                        "--no-enable-shared-context", "--offline"]}

    def test_deployment_matches_frozen_budget(self):
        audit.check_deployment(self.deployment, "d3-mini", self.config, "frozen-image")

    def test_resource_model_and_environment_drift_fail(self):
        for key, value in (("tp", 2), ("cpu_budget", 16), ("host_memory_limit_gib", 128),
                           ("served_model", "d3-flash"), ("image_id", "another-image"),
                           ("env", {"OMP_NUM_THREADS": "1"})):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                audit.check_deployment({**self.deployment, key: value}, "d3-mini", self.config, "frozen-image")

    def test_cache_or_ambiguous_duplicate_flags_fail(self):
        for flag in ("--result-cache-entries", "--threads"):
            d = deepcopy(self.deployment)
            d["command"] += [flag, "4"]
            with self.subTest(flag=flag), self.assertRaises(AssertionError):
                audit.check_deployment(d, "d3-mini", self.config, "frozen-image")

    def test_restoration_requires_identity_not_just_checked_flag(self):
        original = [{"id": str(i), "name": str(i), "image": "fixed", "command": ["serve"],
                     "running": True, "oom": False, "pid": i+1, "restart_count": 0} for i in range(2)]
        restoration = {"checked": True, "before": original, "after": deepcopy(original)}
        audit.check_restoration(original, restoration)
        for field, value in (("id", "replacement"), ("oom", True), ("running", False), ("restart_count", 1)):
            changed = deepcopy(restoration)
            changed["after"][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit.check_restoration(original, changed)


if __name__ == "__main__":
    unittest.main()
