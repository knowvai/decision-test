"""Fetch pinned public corpus ranges, then verify the full official LFS SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import ssl
import time
from urllib.request import Request, urlopen

URL = "https://huggingface.co/datasets/THUIR/T2Ranking/resolve/2a369a430a70979223f1b9a41b1919774d46b432/data/collection.tsv"
SIZE = 3659243528
SHA256 = "07b84e543e9ba696124a727c00629d5bce586631648c436c98dd6e9b146da212"
CHUNK = 4 * 1024 * 1024


def main(args):
    args.parts.mkdir(exist_ok=True, parents=True)
    assert not args.output.exists()
    context = ssl.create_default_context(cafile="/etc/ssl/cert.pem")

    def fetch(index):
        start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
        part = args.parts / f"{index:05d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return index
        for attempt in range(4):
            try:
                request = Request(URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                with urlopen(request, timeout=60, context=context) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"] == f"bytes {start}-{end}/{SIZE}"
                    with part.with_suffix(".tmp").open("wb") as stream:
                        remaining = end - start + 1
                        while remaining:
                            data = response.read(min(1024 * 1024, remaining))
                            if not data:
                                raise RuntimeError("Incomplete range")
                            stream.write(data)
                            remaining -= len(data)
                        assert not response.read(1)
                part.with_suffix(".tmp").replace(part)
                return index
            except Exception:
                if attempt == 3:
                    raise RuntimeError(f"Corpus range {index} failed after four attempts") from None
                time.sleep(2 ** attempt)

    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=12) as pool:
        done = 0
        for future in as_completed([pool.submit(fetch, i) for i in range(count)]):
            future.result()
            done += 1
            if done % 20 == 0 or done == count:
                print(json.dumps({"complete_ranges": done, "total_ranges": count}), flush=True)
    digest = hashlib.sha256()
    with args.output.with_suffix(".assembling").open("xb") as out:
        for index in range(count):
            with (args.parts / f"{index:05d}.part").open("rb") as stream:
                for data in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(data)
                    out.write(data)
    assert digest.hexdigest() == SHA256
    assert args.output.with_suffix(".assembling").stat().st_size == SIZE
    args.output.with_suffix(".assembling").rename(args.output)
    print(json.dumps({"verified": True, "bytes": SIZE, "sha256": SHA256}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
