import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).parents[1] / "decision-eval/patch_sglang_length.py"
spec = importlib.util.spec_from_file_location("sglang_length_guard", path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_unknown_upstream_revision_fails_closed():
    with pytest.raises(AssertionError, match="revision"):
        guard.patch("unexpected upstream source")


def test_guard_preserves_full_valid_input_and_refuses_overflow(monkeypatch):
    source = "def encode(request, max_length):\n    fixed_length = 2\n" + guard.OLD + "\n    return state\n"
    monkeypatch.setattr(guard, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    namespace = {"tokens": list, "_render": str}
    exec(guard.patch(source), namespace)
    assert namespace["encode"](SimpleNamespace(state="abc"), 5) == list("abc")
    with pytest.raises(ValueError, match="Complete input"):
        namespace["encode"](SimpleNamespace(state="abcd"), 5)
