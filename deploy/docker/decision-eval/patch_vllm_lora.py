"""Opt out of pinned CPU LoRA buffers in the pinned evaluation runtime."""

import hashlib
import importlib.util
from pathlib import Path


SOURCE_SHA256 = {
    "lora_model.py": "57d0c10b374a29e2b4fda826dfd9657712394c97ddf6a8099783f718ae2860a4",
    "lora_weights.py": "767c82ea9d3a302f14046af43c638d41389a6b5136b02b08ead6ab9128b9191c",
    "model_manager.py": "4984372726582b9ef10756d4f859e975d1c813116f373bfd8826d9be937a1e3b",
}
IMPORT = "from vllm.utils.torch_utils import PIN_MEMORY"
OLD = 'pin_memory = str(device) == "cpu" and PIN_MEMORY'
NEW = (OLD + '\n        pin_memory = pin_memory and '
       '_lora_transfer_environ.get("VLLM_LORA_DISABLE_PIN_MEMORY", "0") != "1"')
ACTIVATE_OLD = """            module.set_lora(
                index,
                module_lora.lora_a,
                module_lora.lora_b,
            )"""
ACTIVATE_NEW = """            lora_a, lora_b = module_lora.lora_a, module_lora.lora_b
            if _lora_transfer_environ.get("VLLM_LORA_VERIFY_TRANSFER", "0") == "1":
                lora_a = _verified_lora_transfer(lora_a, self.device)
                lora_b = _verified_lora_transfer(lora_b, self.device)
            module.set_lora(index, lora_a, lora_b)"""
HELPER = '''

def _verified_lora_transfer(value, device):
    """Snapshot and verify adapter transport before native layer installation."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return type(value)(_verified_lora_transfer(v, device) for v in value)
    snapshot = torch.empty_like(value, device="cpu", pin_memory=False,
                                memory_format=torch.contiguous_format)
    snapshot.copy_(value, non_blocking=False)
    if not bool(torch.isfinite(snapshot).all()):
        raise ValueError("Nonfinite source LoRA weights")
    uploaded = snapshot.to(device=device, non_blocking=False)
    if not torch.equal(uploaded.cpu(), snapshot):
        raise RuntimeError("LoRA host/device transfer mismatch")
    return uploaded
'''


def patch(source, filename):
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256[filename], \
        "Unexpected vLLM LoRA loader revision"
    if filename == "model_manager.py":
        assert source.count(ACTIVATE_OLD) == source.count("import torch\n") == 1
        result = source.replace("import torch\n", "from os import environ as _lora_transfer_environ\nimport torch\n" + HELPER)
        result = result.replace(ACTIVATE_OLD, ACTIVATE_NEW)
    else:
        assert source.count(OLD) == source.count(IMPORT) == 1
        result = source.replace(IMPORT, "from os import environ as _lora_transfer_environ\n" + IMPORT)
        result = result.replace(OLD, NEW)
    compile(result, filename, "exec")
    return result


if __name__ == "__main__":
    package = Path(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))
    patched = {package / "lora" / name: patch((package / "lora" / name).read_text(), name)
               for name in SOURCE_SHA256}
    for target, source in patched.items():
        target.write_text(source)
    print("Opt-in pageable LoRA loading installed; forward and scoring unchanged.")
