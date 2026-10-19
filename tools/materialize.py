"""Copy code/reports and extract the public archives into a NEW working directory."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("evaluation", "deploy", "docs", "datasets", "tools", "manifests"):
        shutil.copytree(ROOT / name, output / name)
    for path in (output / "docs").rglob("*.json.gz"):
        path.with_suffix("").write_bytes(gzip.decompress(path.read_bytes()))
    manifest = json.loads((ROOT / "manifests/publication.json").read_text())
    for entry in manifest["archives"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
        destination = output / "evidence" / entry["extract_under"]
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                assert member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(tar.extractfile(member).read())
    print(json.dumps({"output": str(output), "archives": len(manifest["archives"]),
                      "corpus_text_included": False, "inference_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
