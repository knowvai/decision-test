"""Make the pinned Clef encoder reject overflow rather than shorten benchmark input."""

import hashlib
import importlib.util
from pathlib import Path


SOURCE_SHA256 = "965c5fc09c3a9e6ca31b5fd9a059922b14943ef929ba901c2aa95a025ca7a0a0"
OLD = "    state = tokens(_render(request.state))[: max_length - fixed_length]"
NEW = (
    "    state = tokens(_render(request.state))\n"
    "    if fixed_length + len(state) > max_length:\n"
    '        raise ValueError("Complete input exceeds the evaluation token limit")'
)


def patch(source):
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA256, "Unexpected SGLang encoder revision"
    assert source.count(OLD) == 1
    return source.replace(OLD, NEW).replace(
        "and\n    the state is truncated so that the expanded prompt has at most",
        "and\n    the complete state is rejected if the expanded prompt exceeds",
    )


if __name__ == "__main__":
    package = Path(next(iter(importlib.util.find_spec("sglang").submodule_search_locations)))
    target = package / "srt/entrypoints/systemone/joint_schema.py"
    target.write_text(patch(target.read_text()))
    print("Strict complete-input guard installed; model and scoring code unchanged.")
