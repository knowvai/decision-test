"""Check archive publication and reconstruction without network or model calls."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_recipe_roundtrip_preserves_unicode_line_separator(tmp_path):
    exporter, reconstructor = module("export_bundle"), module("reconstruct_inputs")
    text = "first\u2028second"
    rows = [{"case_id": "test:1", "query": text, "fragments": [{"text": "fragment"}]}]
    raw = (json.dumps(rows[0], ensure_ascii=False, sort_keys=True) + "\n").encode()
    pub = exporter.Publisher(tmp_path, {})
    pub.recipe(Path("inputs.jsonl"), "fixture", Path("inputs.jsonl"), raw)
    recipe = json.loads(gzip.decompress((tmp_path / pub.recipes[0]["recipe"]).read_bytes()))
    assert set(reconstructor.markers(recipe["records"])) == {
        exporter.sha(text.encode()), exporter.sha(b"fragment")
    }
    restored = reconstructor.restore(recipe["records"], {
        exporter.sha(text.encode()): text, exporter.sha(b"fragment"): "fragment"
    })
    result = "".join(json.dumps(row, **recipe["serialization"]) + "\n" for row in restored).encode()
    assert result == raw


def test_jsonl_requires_complete_record():
    verifier = module("verify")
    assert verifier.jsonl(b'{"value":1}\n') == [{"value": 1}]
    with pytest.raises(AssertionError, match="Incomplete JSONL"):
        verifier.jsonl(b'{"value":1}')


@pytest.mark.parametrize("name", [".env", "key.pem", ".nginx.conf", "runtime.log", "weights.safetensors"])
def test_private_or_binary_material_is_withheld(name):
    assert module("export_bundle").withheld(Path(name)) is not None


def test_added_license_has_explicit_provenance(tmp_path):
    pub = module("export_bundle").Publisher(tmp_path, {})
    pub.record("preflight-20261015", Path("vela-source/LICENSE"), b"license", b"license")
    assert pub.files[0]["provenance"]["kind"] == "publication_supplement"


def archive_fixture(tmp_path, member_name="result.json", symlink=False, bad_hash=False):
    root = tmp_path / "bundle"
    for name in ("evaluation", "deploy", "docs", "datasets", "tools", "manifests"):
        (root / name).mkdir(parents=True)
    path = root / "fixture.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo(member_name)
        if symlink:
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../../outside"
            archive.addfile(entry)
        else:
            entry.size = 3
            archive.addfile(entry, io.BytesIO(b"{}\n"))
    checksum = "0" * 64 if bad_hash else hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifests/publication.json").write_text(json.dumps({"archives": [{
        "path": path.name, "sha256": checksum, "extract_under": "fixture"
    }]}))
    return root


@pytest.mark.parametrize("name,symlink,bad_hash", [
    ("../../outside", False, False), ("/absolute", False, False),
    ("result.json", True, False), ("result.json", False, True),
])
def test_materialize_rejects_unsafe_or_changed_archive(tmp_path, name, symlink, bad_hash):
    materializer = module("materialize")
    materializer.ROOT = archive_fixture(tmp_path, name, symlink, bad_hash)
    with pytest.raises(AssertionError):
        materializer.main(tmp_path / "output")
    assert not (tmp_path / "outside").exists()


def test_materialize_requires_new_output(tmp_path):
    materializer = module("materialize")
    materializer.ROOT = archive_fixture(tmp_path)
    output = tmp_path / "output"
    materializer.main(output)
    assert (output / "evidence/fixture/result.json").read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        materializer.main(output)


def test_downloader_verifies_existing_files_without_network(tmp_path, monkeypatch):
    downloader = module("download_data")
    downloader.ROOT = tmp_path
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/sources.json").write_text(json.dumps({"fixture": {
        "filename": "data.txt", "sha256": hashlib.sha256(b"data").hexdigest()
    }}))
    output = tmp_path / "downloads"
    output.mkdir()
    (output / "data.txt").write_bytes(b"data")
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: pytest.fail("Unexpected network call"))
    args = SimpleNamespace(all=False, sources=["fixture"], output=output)
    downloader.main(args)
    (output / "data.txt").write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Existing file differs"):
        downloader.main(args)


def test_namespace_preserves_hash_and_random_material():
    rules = module("export_bundle").NamespaceRules({
        "replacements": {"private_suite": "evaluation"},
        "seed_aliases": {"private-fixed-v1": "fixed-v1"},
    })
    source = '''import hashlib, json, random
SEED = "private-fixed-v1"
def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()
def sample():
    return digest([SEED, "case"]), hashlib.sha256((SEED + ":case").encode()).hexdigest(), random.Random("private-fixed-v1").random()
protocol = {"seed": SEED, "module": "private_suite"}
'''
    transformed = rules.content(source.encode(), "runner.py")
    assert b"private_suite" not in transformed and b"private-fixed-v1" not in transformed
    old, new = {}, {}
    exec(source, old)
    exec(transformed, new)
    assert old["sample"]() == new["sample"]()
    assert new["protocol"] == {"seed": "fixed-v1", "module": "evaluation"}
    assert rules.content(transformed, "runner.py") == transformed


def test_namespace_updates_paths_without_rewriting_other_values():
    rules = module("export_bundle").NamespaceRules({"replacements": {"private_suite": "evaluation"}})
    raw = b'{"path":"private_suite/results.json","score":0.75,"sha256":"1234"}\n'
    assert json.loads(rules.content(raw, "results.json")) == {
        "path": "evaluation/results.json", "score": 0.75, "sha256": "1234"
    }
