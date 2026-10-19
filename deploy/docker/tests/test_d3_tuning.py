"""D3 pilot isolation, native wire and opt-in launch regression tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("d3_tuning", ROOT / "docs/research/2026-10-19-d3-tuning.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
import launch_reviewed as launcher


class PilotTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_line_separators(self):
        rows = [{"text": "甲\u2028乙\u2029丙\u0085丁"}, {"text": "second\nline"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            t.save_rows(path, rows)
            self.assertEqual(t.read_rows(path), rows)

    def test_wire_excludes_labels_and_requests_full_input(self):
        for task, choices in t.variants().items():
            for question in choices.values():
                job = t.make_job("private-id", "fragment", {"query": "data"}, gold="secret-label")
                path, body = t.wire("d3-mini", job, question)
                self.assertEqual(path, "/v1/systemone")
                self.assertEqual(set(body), {"model", "state", "questions"})
                self.assertTrue(next(iter(body["questions"].values()))["require_full_input"])
                self.assertNotIn("secret-label", json.dumps(body))
                self.assertNotIn("private-id", json.dumps(body))

    def test_threshold_only_changes_valid_predictions(self):
        rows = [{"status": "valid", "prediction": "irrelevant", "probabilities": {"relevant": .45}},
                {"status": "failed", "error_type": "TimeoutError"}]
        result = t.threshold_rows(rows, .45)
        self.assertEqual(result[0]["prediction"], "relevant")
        self.assertEqual(result[1], rows[1])
        self.assertEqual(rows[0]["prediction"], "irrelevant")

    def test_quality_selection_never_reads_holdout(self):
        def score(data, task, split, rows, threshold):
            self.assertEqual(split, "dev")
            return rows
        candidates = {"baseline": {"accuracy": .8, "macro_f1": .79},
                      "concise": {"accuracy": .79, "macro_f1": .9},
                      "explicit": {"accuracy": .8, "macro_f1": .79}}
        with patch.object(t, "score", side_effect=lambda d, task, split, rows, threshold=None: score(d,task,split,rows,threshold)):
            self.assertEqual(t.selection(Path("unused"), "classification", candidates)["variant"], "baseline")

    def test_worker_overrides_fail_before_side_effects(self):
        for model, extra in (("d3-mini", {"worker_threads": 0}), ("d3-mini", {"worker_threads": True}),
                             ("d3-mini", {"batch_window_ms": float("nan")}),
                             ("clef-flash", {"worker_threads": 2}), ("d3-mini", {"batch_window_ms": 21})):
            with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, "remove_owned") as remove:
                with self.assertRaises(ValueError):
                    launcher.launch(SimpleNamespace(root=Path(tmp), model=model, **extra))
                remove.assert_not_called()

    def test_worker_explicit_flags_and_historical_default(self):
        for extra in ({}, {"worker_threads": 1, "batch_window_ms": 0}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "models/d3-mini").mkdir(parents=True)
                args = SimpleNamespace(root=root, model="d3-mini", env_file=root / ".env",
                                       native_port=30888, public_port=30889, seqs=32, tokens=32768,
                                       enforce_eager=False, **extra)
                def docker(*parts):
                    return "" if parts[-1] == "{{.Config.User}}" else "sha256:" + "a" * 64
                with patch.object(launcher, "docker", side_effect=docker), patch.object(launcher, "remove_owned"), \
                     patch.object(launcher, "read_key", return_value="A1" * 16), \
                     patch.object(launcher.socket, "socket"), patch.object(launcher.subprocess, "check_output", return_value=""):
                    launcher.launch(args)
                record = json.loads((root / "acceptance-20261015/evidence/d3-mini.deployment.json").read_text())
                cmd = record["command"]
                self.assertEqual(cmd[cmd.index("--threads")+1], str(extra.get("worker_threads", 2)))
                self.assertEqual(record["env"]["OMP_NUM_THREADS"], str(extra.get("worker_threads", 2)))
                self.assertEqual("--batch-window-ms" in cmd, bool(extra))
                self.assertNotIn("DECISION_EVAL_API_KEY", record["env"])


if __name__ == "__main__":
    unittest.main()
