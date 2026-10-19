"""Acceptance must reject malformed native probabilities, not just HTTP errors."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "decision-eval"))
from deployment_probe import native_choice_request, native_choice_result, valid_probabilities


class ProbabilityValidationTests(unittest.TestCase):
    def test_complete_finite_distribution(self):
        self.assertTrue(valid_probabilities({"yes": 0.8, "no": 0.2}, ["yes", "no"]))

    def test_invalid_distributions(self):
        for value in ({"yes": 1}, {"yes": float("nan"), "no": 0},
                      {"yes": 0.4, "no": 0.4}, {"yes": True, "no": False},
                      {"yes": -0.1, "no": 1.1}, [0.8, 0.2]):
            with self.subTest(value=value):
                self.assertFalse(valid_probabilities(value, ["yes", "no"]))

    def test_native_jev_mapping_preserves_selected_index_and_full_state(self):
        payload = {"model": "example", "state": {"utterance": "full input"}, "questions": {
            "domain": {"type": "choice", "instructions": "choose", "criteria": {"a": "alpha", "b": "beta"}}}}
        path, body = native_choice_request("jev", payload)
        self.assertEqual(path, "/v1/decide")
        self.assertEqual(body["options"], ["alpha", "beta"])
        self.assertEqual(body["strategy"], "single")
        self.assertEqual(body["thinking"], "off")
        response = {"options": ["alpha", "beta"], "probabilities": [0.5, 0.5],
                    "choice_index": 1, "choice": "beta"}
        self.assertEqual(native_choice_result("jev", payload, response)[0], "b")
        with self.assertRaises(AssertionError):
            native_choice_result("jev", payload, {**response, "probabilities": [1]})

    def test_worker_requires_full_input_without_mutating_prompt(self):
        payload = {"model": "Vela-test", "state": "full", "questions": {
            "domain": {"type": "choice", "instructions": "choose", "criteria": {"a": "alpha"}}}}
        path, body = native_choice_request("worker", payload)
        self.assertEqual(path, "/v1/systemone")
        self.assertTrue(body["questions"]["domain"]["require_full_input"])
        self.assertNotIn("require_full_input", payload["questions"]["domain"])
        response = {"answers": {"domain": {"type": "choice", "choice": "a", "probabilities": {"a": 1}}}}
        with self.assertRaises(AssertionError):
            native_choice_result("worker", payload, response)
        response["answers"]["domain"]["input_coverage"] = "complete"
        self.assertEqual(native_choice_result("worker", payload, response)[0], "a")


if __name__ == "__main__":
    unittest.main()
