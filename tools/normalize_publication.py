"""Normalize an existing public bundle, including archive members and provenance.

Rules are supplied separately. Original source checksums are never rewritten.
"""
import argparse
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import tarfile

from export_bundle import NamespaceRules, json_bytes, sha


IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", "downloads", "work"}


def safe_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe publication path")
    return path


def normalize(root, rules):
    manifest_path = root / "manifests/publication.json"
    manifest = json.loads(manifest_path.read_text())
    indexed = {(r["archive"], r["path"]): r for r in manifest["files"] if "archive" in r}
    changed_archives = changed_members = 0
    for entry in manifest["archives"]:
        path = root / safe_path(entry["path"])
        before = path.read_bytes()
        if sha(before) != entry["sha256"]:
            raise ValueError("Archive differs from manifest: " + entry["path"])
        buffer, changed, names = io.BytesIO(), False, set()
        with tarfile.open(fileobj=io.BytesIO(before), mode="r:gz") as original:
            with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as output:
                for member in original:
                    if not member.isfile():
                        raise ValueError("Non-file archive member")
                    name = str(safe_path(rules.text(member.name)))
                    if name in names:
                        raise ValueError("Namespace collision: " + name)
                    names.add(name)
                    raw = original.extractfile(member).read()
                    data = rules.content(raw, member.name)
                    row = indexed[(entry["path"], member.name)]
                    if data != raw or name != member.name:
                        changed = True
                        changed_members += 1
                    row.update(path=name, published_bytes=len(data), published_sha256=sha(data))
                    row["publication"] = "unchanged" if sha(data) == row["source_sha256"] else "metadata_redacted"
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), member.mode, member.mtime
                    output.addfile(info, io.BytesIO(data))
        if changed:
            data = gzip.compress(buffer.getvalue(), mtime=0)
            path.write_bytes(data)
            entry.update(bytes=len(data), sha256=sha(data))
            changed_archives += 1

    files = [p for p in root.rglob("*") if p.is_file() and not IGNORED.intersection(p.relative_to(root).parts)]
    changed_files = 0
    for path in files:
        if path == manifest_path or path.name.endswith(".tar.gz"):
            continue
        rel = path.relative_to(root)
        target = root / safe_path(rules.text(str(rel)))
        original = path.read_bytes()
        compressed = path.suffix == ".gz"
        raw = gzip.decompress(original) if compressed else original
        data = rules.content(raw, rel.with_suffix("") if compressed else rel)
        if data != raw or target != path:
            if target != path and target.exists():
                raise ValueError("Namespace collision: " + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.compress(data, mtime=0) if compressed else data)
            if target != path:
                path.unlink()
            changed_files += 1
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not IGNORED.intersection(path.relative_to(root).parts) and not any(path.iterdir()):
            path.rmdir()

    manifest = json.loads(rules.text(json.dumps(manifest, ensure_ascii=False)))
    for row in manifest["files"]:
        if "destination" not in row:
            continue
        raw = (root / safe_path(row["destination"])).read_bytes()
        data = gzip.decompress(raw) if row["gzip"] else raw
        row.update(published_bytes=len(data), published_sha256=sha(data))
        row["publication"] = "unchanged" if sha(data) == row["source_sha256"] else "metadata_redacted"
    manifest["counts"] = dict(Counter(row["publication"] for row in manifest["files"]))
    manifest["substitutions"] = "Host/user metadata and publication namespaces normalized; original source hashes retained; fixed seed material preserved separately."
    manifest_path.write_bytes(json_bytes(manifest))
    if rules.seeds:
        (root / "manifests/seed-material.json").write_bytes(json_bytes({
            "purpose": "Fixed UTF-8 seed bytes used by the historical hash/random calculations; aliases are presentation identifiers only.",
            "seeds": {alias: {"utf8_hex": value.encode().hex()} for value, alias in rules.seeds.items()},
        }))
    return {"changed_archives": changed_archives, "changed_archive_members": changed_members,
            "changed_files": changed_files, "counts": manifest["counts"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rules", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize(args.root, NamespaceRules(json.loads(args.rules.read_text())))))
