"""The appliance workaround changes LoRA loading only, and is explicitly opt-in."""

import ast
import hashlib
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

SOURCE = Path(__file__).parents[1] / "decision-eval"
sys.path.insert(0, str(SOURCE))
import patch_vllm_lora as target


class LoRALoadingPatchTests(unittest.TestCase):
    def fixture(self):
        return target.IMPORT + '\nclass Loader:\n    def load(self, device):\n        ' + target.OLD + '\n        return pin_memory\n'

    def test_refuses_unknown_revision_and_reapplication(self):
        for name in ("lora_model.py", "lora_weights.py"):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                target.patch(self.fixture(), name)
        source = self.fixture()
        with mock_patch.dict(target.SOURCE_SHA256, {"lora_model.py": hashlib.sha256(source.encode()).hexdigest()}):
            result = target.patch(source, "lora_model.py")
            with self.assertRaises(AssertionError):
                target.patch(result, "lora_model.py")

    def test_default_and_opt_in_across_devices(self):
        source = self.fixture()
        for name in ("lora_model.py", "lora_weights.py"):
            with mock_patch.dict(target.SOURCE_SHA256, {name: hashlib.sha256(source.encode()).hexdigest()}):
                result = target.patch(source, name)
            tree = ast.parse(result)
            loader = ast.Module(body=[tree.body[-1]], type_ignores=[])
            for available in (False, True):
                for setting in (None, "0", "1"):
                    for device in ("cpu", "cuda"):
                        env = {} if setting is None else {"VLLM_LORA_DISABLE_PIN_MEMORY": setting}
                        namespace = {"PIN_MEMORY": available, "_lora_transfer_environ": env}
                        exec(compile(loader, "fixture", "exec"), namespace)
                        expected = device == "cpu" and available and setting != "1"
                        self.assertEqual(namespace["Loader"]().load(device), expected)
            self.assertEqual(result.count(target.NEW), 1)

    def test_activation_patch_is_scoped_and_hash_checked(self):
        source = "import torch\nclass Manager:\n    def activate(self):\n" + target.ACTIVATE_OLD + "\n"
        with self.assertRaises(AssertionError):
            target.patch(source, "model_manager.py")
        with mock_patch.dict(target.SOURCE_SHA256, {"model_manager.py": hashlib.sha256(source.encode()).hexdigest()}):
            result = target.patch(source, "model_manager.py")
        self.assertEqual(result.count(target.ACTIVATE_NEW), 1)
        self.assertIn('"VLLM_LORA_VERIFY_TRANSFER", "0"', result)
        self.assertEqual(result.count("def _verified_lora_transfer"), 1)

    def test_verified_transfer_preserves_values_and_rejects_corruption(self):
        class Tensor:
            corrupt = False

            def __init__(self, values):
                self.values = list(values)

            def copy_(self, source, non_blocking):
                self.values = list(source.values)

            def to(self, device, non_blocking):
                return Tensor([999] if self.corrupt else self.values)

            def cpu(self):
                return self

        allocations = []

        def empty_like(value, **kwargs):
            allocations.append(kwargs)
            return Tensor([])

        torch = SimpleNamespace(empty_like=empty_like, contiguous_format="contiguous",
            isfinite=lambda t: SimpleNamespace(all=lambda: all(math.isfinite(v) for v in t.values)),
            equal=lambda a, b: a.values == b.values)
        namespace = {"torch": torch}
        exec(target.HELPER, namespace)
        transfer = namespace["_verified_lora_transfer"]
        source = Tensor([1, 2, 3])
        result = transfer([source, None, (source,)], "cuda:1")
        self.assertEqual(result[0].values, source.values)
        self.assertIsNot(result[0], source)
        self.assertIsNone(result[1])
        self.assertIsInstance(result[2], tuple)
        self.assertTrue(all(a["device"] == "cpu" and a["pin_memory"] is False for a in allocations))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            transfer(Tensor([float("nan")]), "cuda:1")
        Tensor.corrupt = True
        with self.assertRaisesRegex(RuntimeError, "transfer mismatch"):
            transfer(source, "cuda:1")
        self.assertEqual(source.values, [1, 2, 3])

    def test_activation_uses_manager_device_for_layers_without_device_attribute(self):
        source = ("import torch\nclass Manager:\n"
                  "    def activate(self, module, module_lora, index):\n" + target.ACTIVATE_OLD + "\n")
        with mock_patch.dict(target.SOURCE_SHA256, {"model_manager.py": hashlib.sha256(source.encode()).hexdigest()}):
            result = target.patch(source, "model_manager.py")
        manager_node = next(node for node in ast.parse(result).body if isinstance(node, ast.ClassDef))
        for enabled in (False, True):
            transfers, installed = [], []

            def transfer(value, device):
                transfers.append(device)
                return ("uploaded", value)

            namespace = {"_lora_transfer_environ": {"VLLM_LORA_VERIFY_TRANSFER": "1" if enabled else "0"},
                         "_verified_lora_transfer": transfer}
            exec(compile(ast.Module(body=[manager_node], type_ignores=[]), "fixture", "exec"), namespace)
            manager = namespace["Manager"]()
            manager.device = "cuda:1"
            module = SimpleNamespace(set_lora=lambda *args: installed.append(args))
            manager.activate(module, SimpleNamespace(lora_a="a", lora_b="b"), 3)
            self.assertEqual(transfers, ["cuda:1", "cuda:1"] if enabled else [])
            self.assertEqual(installed, [(3, ("uploaded", "a"), ("uploaded", "b"))] if enabled else [(3, "a", "b")])


if __name__ == "__main__":
    unittest.main()
