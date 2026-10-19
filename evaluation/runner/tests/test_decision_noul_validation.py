"""Offline mode-contract and matched-denominator checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[3] / "deploy/docker/decision-eval"))
import decision_noul_validation as n
from deployment_probe import native_question_request, native_noul_result
from test_decision_passage_benchmark import pool, labels


class NoulTests(unittest.TestCase):
    def test_only_mode_and_criteria_change(self):
        for key, q in n.p.QUESTIONS.items():
            self.assertEqual(n.QUESTIONS[key], {"type": "noul", "instructions": q["instructions"]})
        self.assertEqual(len(n.MODELS), 13)
        self.assertFalse(any("qwen" in model for model in n.MODELS))

    def test_complete_input_mapping_and_jev_nonthinking(self):
        body = n.payload("model", {"query": "q", "document": "d"})
        before = deepcopy(body)
        path, wire = native_question_request("worker", body)
        self.assertTrue(wire["questions"]["relevance"]["require_full_input"])
        self.assertEqual(body, before)
        _, wire = native_question_request("laya", body)
        self.assertEqual(wire["max_len"], 8192)
        self.assertEqual(wire["head_max_len"], 8192)
        path, wire = native_question_request("jev", body)
        self.assertEqual(path, "/v1/decide")
        self.assertEqual(wire["kind"], "noul")
        self.assertEqual(wire["thinking"], "off")
        self.assertEqual(wire["strategy"], "single")
        self.assertNotIn("options", wire)
        self.assertEqual(json.loads(wire["state"]), body["state"])

    def test_probability_binary_readout_and_tie(self):
        for value, expected in ((.4, "irrelevant"), (.5, "irrelevant"), (.50001, "relevant")):
            raw = {"answers": {"relevance": {"type": "noul", "noul": value}}}
            result = native_noul_result("sglang", n.payload("model", {}), raw)
            self.assertEqual(result["prediction"], expected)
            self.assertEqual(result["noul"], value)
            self.assertIsNone(result["native_choice"])
            self.assertEqual(native_noul_result("sglang", n.payload("model", {}), result["native_response"])["prediction"], expected)

    def test_native_boolean_preserved(self):
        for value in (True, False):
            raw = {"answers": {"relevance": {"type": "noul", "noul": value}}}
            result = native_noul_result("sdk", n.payload("model", {}), raw)
            self.assertEqual(result["readout"], "native_boolean")
            self.assertEqual(result["prediction"], "relevant" if value else "irrelevant")

    def test_raw_distribution_and_coverage(self):
        raw = {"answers": {"relevance": {"type": "noul", "noul": .7,
              "probabilities": {"true": .7, "false": .3}, "input_coverage": "complete"}}}
        native_noul_result("worker", n.payload("Vela-test", {}), raw)
        for key, value in (("type", "choice"), ("noul", float("nan")), ("noul", 1.1),
                           ("noul", "true"), ("probabilities", {"true": .2, "false": .8}), ("input_coverage", "partial")):
            invalid = deepcopy(raw)
            invalid["answers"]["relevance"][key] = value
            with self.assertRaises(AssertionError):
                native_noul_result("worker", n.payload("Vela-test", {}), invalid)

    def test_truncation_and_fallback_rejected(self):
        raw = {"answers": {"relevance": {"type": "noul", "noul": .7}}}
        for field, key in (("usage", "truncated"), ("usage", "truncated_tokens"),
                           ("execution", "truncated"), ("execution", "cpu_fallback")):
            with self.assertRaises(AssertionError):
                native_noul_result("sdk", n.payload("model", {}), {**raw, field: {key: 1}})

    def test_jev_native_category_and_reject_other_modes(self):
        raw = {"kind": "noul", "effective_kind": "noul", "adaptation": "native",
               "options": ["false", "true"], "probabilities": [.5, .5], "choice_index": 0,
               "choice": "false", "num_model_requests": 1}
        result = native_noul_result("jev", n.payload("model", {}), raw)
        self.assertEqual(result["prediction"], "irrelevant")
        self.assertEqual(result["readout"], "native_choice")
        for key, value in (("kind", "choice"), ("options", ["true", "false"]),
                           ("num_model_requests", 2), ("thinking", {"used": True})):
            with self.assertRaises(AssertionError):
                native_noul_result("jev", n.payload("model", {}), {**raw, key: value})

    def test_failed_and_missing_pairs_stay_in_denominator(self):
        data = pool(3)
        rows = [{"case_id": data["case_id"], "fragment_id": data["candidates"][0]["fragment_id"],
                 "status": "valid", "prediction": "relevant"}]
        result = n.quality([data], [labels(data)], rows)
        self.assertEqual(result["candidate_accuracy"], 1/3)
        self.assertEqual(result["counts"]["unscored"], 2)
        self.assertEqual(result["quality"]["f1"]["point"], 0)
        self.assertEqual(result["valid_pools"], 0)

    def test_matched_audit_reconstructs_hashes_and_native_readout(self):
        from launch_reviewed import MODELS
        model = "d3-edge"
        family, _, served = MODELS[model]
        data = pool(2)
        rows, intents = [], []
        for i, candidate in enumerate(data["candidates"]):
            identity = {"model": model, "phase": "validation", "arm": "noul", "stage": "refine",
                        "case_id": data["case_id"], "fragment_id": candidate["fragment_id"]}
            state = n.p.pair_state(data, candidate)
            body = n.payload(served, state)
            path, wire = native_question_request(family, body)
            parsed = native_noul_result(family, body, {"answers": {"relevance": {"type": "noul", "noul": .8 if i == 0 else .2}}})
            row = {**identity, **parsed, "call_id": n.b.digest(identity), "request_sha256": n.b.digest(wire),
                   "input_sha256": n.b.digest(state), "path": path, "calls": 1, "status": "valid",
                   "http_status": 200, "inflight_at_start": 1, "started_seconds": i*2,
                   "http_started_seconds": i*2+.1, "ended_seconds": i*2+1, "elapsed_ms": 900}
            rows.append(row)
            intents.append(dict(row))
        with tempfile.TemporaryDirectory() as tmp:
            folder, baseline = Path(tmp)/"new", Path(tmp)/"old"
            folder.mkdir()
            (baseline/model).mkdir(parents=True)
            n.p.save_rows(folder/"requests.jsonl", rows)
            n.p.save_rows(folder/"intents.jsonl", intents)
            n.p.save_rows(baseline/model/"requests.jsonl", rows)
            result = n.audit_model(folder, model, [data], [labels(data)], baseline)
            self.assertTrue(result["passed"])
            self.assertEqual(result["noul"]["quality"]["f1"]["point"], 1)
            self.assertEqual(result["flips"], 0)
            with patch.object(n.b, "read_jsonl", side_effect=[[{**rows[0], "prediction": "irrelevant"}, rows[1]], intents, rows]):
                with self.assertRaises(AssertionError):
                    n.audit_model(folder, model, [data], [labels(data)], baseline)


if __name__ == "__main__":
    unittest.main()
