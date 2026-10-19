"""Restore frozen inputs from source-text hashes; never calls a model or changes labels."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def markers(value):
    if isinstance(value, dict):
        if set(value) == {"__source_text_sha256__"}:
            yield value["__source_text_sha256__"]
        else:
            for child in value.values():
                yield from markers(child)
    elif isinstance(value, list):
        for child in value:
            yield from markers(child)


def restore(value, texts):
    if isinstance(value, dict):
        if set(value) == {"__source_text_sha256__"}:
            return texts[value["__source_text_sha256__"]]
        return {k: restore(v, texts) for k, v in value.items()}
    if isinstance(value, list):
        return [restore(v, texts) for v in value]
    return value


def main(args):
    import pyarrow.parquet as pq
    manifest = json.loads((ROOT / "manifests/publication.json").read_text())
    recipes = []
    for row in manifest["dataset_recipes"]:
        if not args.experiment or row["source"] in args.experiment:
            recipes.append(json.loads(gzip.decompress((ROOT / row["recipe"]).read_bytes())))
    assert recipes, "No matching recipes"
    wanted = set(x for recipe in recipes for x in markers(recipe["records"]))
    texts = {}

    def remember(value):
        key = sha(value.encode())
        if key in wanted:
            texts[key] = value

    sources = json.loads((ROOT / "datasets/sources.json").read_text())
    used = []
    for key, record in sources.items():
        paths = [directory / name for directory in args.sources_dir
                 for name in [record["filename"], *record.get("aliases", [])]]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            continue
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"], key
        used.append(key)
        if record["format"] in {"massive", "t2reranking"}:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    if record["format"] == "massive":
                        remember(row["utt"])
                    else:
                        for text in [row["query"], *row["positive"], *row["negative"]]:
                            remember(text)
        elif record["format"] == "tsv-text":
            csv.field_size_limit(1024 * 1024 * 1024)
            with path.open() as stream:
                for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
                    remember(row["text"])
    missing = wanted - texts.keys()
    if missing:
        affected = [{"experiment": r["source"], "path": r["path"],
                     "missing_texts": len(set(markers(r["records"])) & missing)}
                    for r in recipes if set(markers(r["records"])) & missing]
        raise ValueError(json.dumps({"missing_texts": len(missing), "affected": affected,
                                     "message": "Supply pinned sources. No inputs written."}))
    restored = []
    for recipe in recipes:
        records = restore(recipe["records"], texts)
        raw = "".join(json.dumps(row, **recipe["serialization"]) + "\n" for row in records).encode()
        assert sha(raw) == recipe["sha256"] and len(raw) == recipe["bytes"], recipe["path"]
        path = args.output / recipe["source"] / recipe["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == raw, f"Refusing to replace: {path}"
        else:
            path.write_bytes(raw)
        restored.append({"experiment": recipe["source"], "path": recipe["path"],
                         "records": len(records), "sha256": recipe["sha256"]})
    result = {"passed": True, "files": restored, "sources_verified": used,
              "unique_texts": len(texts), "inference_calls": 0}
    if args.report:
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"passed": True, "files": len(restored), "sources": used, "inference_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", type=Path, action="append", required=True)
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    main(parser.parse_args())
