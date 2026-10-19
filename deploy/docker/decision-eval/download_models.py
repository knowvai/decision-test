"""Download ModelScope mirrors and materialize hash-verified official releases."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from modelscope import snapshot_download


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


root = Path(os.environ["EVAL_ROOT"])
parser = argparse.ArgumentParser()
parser.add_argument("--lock", type=Path, default=root / "source" / "model-lock.json")
parser.add_argument("--model")
args = parser.parse_args()
lock = json.loads(args.lock.read_text())
if args.model and args.model not in {p["directory"] for p in lock["models"]}:
    parser.error("Model is not present in the selected file lock")
for package in lock["models"]:
    if args.model and package["directory"] != args.model:
        continue
    files = package["files_sha256"]
    snapshot = Path(snapshot_download(package["modelscope_id"], revision=package.get("modelscope_revision", "master"), cache_dir=str(root / "cache" / "modelscope"), allow_patterns=list(files), max_workers=int(os.environ.get("MODELSCOPE_FILE_WORKERS", "4")), endpoint=os.environ.get("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")))
    target = root / "models" / package["directory"]
    target.mkdir(parents=True, exist_ok=True)
    restored = []
    for name, expected in files.items():
        source = snapshot / name
        if not source.is_file() or digest(source) != expected:
            source = args.lock.parent / "official-files" / package["directory"] / name
            if not source.is_file() or digest(source) != expected:
                raise RuntimeError(f"Mirror mismatch without verified original: {package['modelscope_id']}/{name}")
            restored.append(name)
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != expected:
                raise RuntimeError(f"Existing release file differs: {destination}")
        else:
            os.link(source.resolve(), destination)
        print(f"VERIFIED {package['directory']}/{name}", flush=True)
    record = {**package, "restored_official_files": restored, "source": "ModelScope, with pinned official originals for mirror differences"}
    (root / "logs" / (package["directory"] + "-provenance.json")).write_text(json.dumps(record, indent=2) + "\n")
