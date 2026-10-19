"""Download explicitly selected, pinned public datasets; verify SHA256 before use."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(args):
    sources = json.loads((ROOT / "datasets/sources.json").read_text())
    names = list(sources) if args.all else args.sources
    if not names:
        print(json.dumps({k: {"filename": v["filename"], "license": v["license"]} for k, v in sources.items()}, indent=2))
        return
    assert set(names) <= sources.keys(), "Unknown source"
    args.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        record = sources[name]
        path = args.output / record["filename"]
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            request = Request(record["url"], headers={"Accept-Encoding": "identity"})
            with urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            assert digest(partial) == record["sha256"], f"Hash mismatch: {name}; partial retained"
            partial.rename(path)
        assert digest(path) == record["sha256"], f"Existing file differs: {name}"
        print(json.dumps({"source": name, "sha256_verified": True, "bytes": path.stat().st_size}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "downloads")
    main(parser.parse_args())
