"""Jeva transport mapping without GPU weights or model decisions."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "jeva_backend", Path(__file__).parents[1] / "decision-eval" / "jeva_backend.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class JevaMappingTests(unittest.TestCase):
    def runtime(self, truncated=0):
        answers = {
            "route": SimpleNamespace(kind="choice", choice="b", probabilities={"a": .2, "b": .8}, confidence=.8),
            "truth": SimpleNamespace(kind="noul", noul=.7, probabilities={"false": .3, "true": .7}, confidence=.7),
            "level": SimpleNamespace(kind="score", score=.6, probabilities={"low": .4, "high": .6}, confidence=.6),
        }
        return SimpleNamespace(cache="none", option_chunk_size=16, option_batch_tokens=16384,
            system_one=Mock(return_value=SimpleNamespace(answers=answers,
                usage=SimpleNamespace(to_dict=lambda: {"input_tokens": 11, "output_tokens": 0, "truncated_tokens": truncated}))))

    def test_preserves_inputs_and_maps_typed_outputs(self):
        sdk = SimpleNamespace(Choice=Mock(), Noul=Mock(), Score=Mock())
        body = {"model": "fixture", "state": {"text": "unchanged"}, "questions": {
            "route": {"type": "choice", "instructions": "choose", "criteria": {"a": "first", "b": "second"}},
            "truth": {"type": "noul", "instructions": "verify"},
            "level": {"type": "score", "instructions": "assess", "criteria": ["low", "high"]}}}
        runtime = self.runtime()
        with patch.dict(sys.modules, {"jeva": sdk}):
            result = module.predict_jeva(runtime, body)
        sdk.Choice.assert_called_once_with("choose", {"a": "first", "b": "second"}, allow_null=False)
        sdk.Score.assert_called_once_with("assess", ["low", "high"], values=[0, 1])
        self.assertIs(runtime.system_one.call_args.kwargs["state"], body["state"])
        self.assertEqual(result["answers"]["route"]["choice"], "b")
        self.assertEqual(result["answers"]["truth"]["noul"], .7)
        self.assertEqual(result["answers"]["level"]["probabilities"], {"0": .4, "1": .6})
        self.assertEqual(result["answers"]["level"]["score"], .6)
        self.assertFalse(result["execution"]["cross_request_batching"])

    def test_missing_answers_and_truncation_fail(self):
        sdk = SimpleNamespace(Choice=Mock(), Noul=Mock(), Score=Mock())
        body = {"model": "fixture", "state": "data", "questions": {
            "route": {"type": "noul", "instructions": "verify"}}}
        with patch.dict(sys.modules, {"jeva": sdk}):
            with self.assertRaises(RuntimeError):
                module.predict_jeva(self.runtime(), body)
            runtime = self.runtime(truncated=1)
            runtime.system_one.return_value.answers = {"route": runtime.system_one.return_value.answers["truth"]}
            with self.assertRaises(RuntimeError):
                module.predict_jeva(runtime, body)


if __name__ == "__main__":
    unittest.main()
