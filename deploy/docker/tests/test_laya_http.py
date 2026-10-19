"""The official HTTP app reuses the same strict input boundary as the SDK path."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class LayaHttpGuardTests(unittest.TestCase):
    def test_guard_checks_each_state_with_actual_state_and_head_budgets(self):
        validate = Mock()
        spec = importlib.util.spec_from_file_location("laya_http", Path(__file__).parents[1] / "decision-eval/laya_http.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"server": SimpleNamespace(validate_laya_input=validate)}):
            spec.loader.exec_module(module)
        agent = SimpleNamespace(cfg={"max_len": 1024, "head_max_len": 256})
        ctx = SimpleNamespace(agent=agent, states=["first", "second"], questions={"q": {}},
                              max_len=8192, head_max_len=512)
        module.guard_input(ctx)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(validate.call_args.args, (agent, {"state": "second", "questions": ctx.questions}, 8192))
        self.assertEqual(validate.call_args.kwargs, {"head_max_length": 512})
        ctx.max_len = ctx.head_max_len = None
        module.guard_input(ctx)
        self.assertEqual(validate.call_args.args[-1], 1024)
        self.assertEqual(validate.call_args.kwargs, {"head_max_length": 256})
        validate.side_effect = ValueError("would truncate")
        with self.assertRaises(ValueError):
            module.guard_input(ctx)


if __name__ == "__main__":
    unittest.main()
