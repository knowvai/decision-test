"""Hash-bound, offline-only explanation of tails and decision flips, without reruns."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offline(event, args):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
        raise RuntimeError("Offline diagnosis forbids network/subprocess")


def main(args):
    sys.addaudithook(offline)
    audit = json.loads(args.audit.read_text())
    assert audit["passed"]
    sources = {}

    def rows(relative):
        path = args.root / relative
        actual = sha(path)
        assert actual == audit["artifact_sha256"][relative], relative
        sources[relative] = actual
        with path.open() as stream:
            return [json.loads(line) for line in stream]

    data = args.root.parent / "d3-tuning-20261019/dataset"
    protocol = json.loads((data/"protocol.json").read_text())
    assert sha(data/"protocol.json") == audit["dataset_sha256"]
    gold = {}
    for task in ("classification", "evidence"):
        path = data/f"{task}-dev.private.jsonl"
        assert sha(path) == protocol["files"][path.name]
        with path.open() as stream:
            gold[task] = {(j["case_id"], j["fragment_id"]): j["gold"] for j in map(json.loads, stream)}
    result = {"models": {}}
    for model, tested in audit["models"].items():
        changes, slow_blocks = defaultdict(list), []
        for profile in audit["profiles"]:
            for task in ("classification", "evidence"):
                for repeat in (1, 2):
                    ref = {(j["case_id"], j["fragment_id"]): j for j in rows(f"{model}/{profile}/{task}-n1-r{repeat}.jsonl")}
                    for n in (8, 16, 32):
                        relative = f"{model}/{profile}/{task}-n{n}-r{repeat}.jsonl"
                        records = rows(relative)
                        for row in records:
                            key = row["case_id"], row["fragment_id"]
                            before = ref[key]
                            if before["prediction"] != row["prediction"]:
                                changes[task, *key].append({"profile": profile, "n": n, "repeat": repeat,
                                    "gold": gold[task][key], "before": before["prediction"], "after": row["prediction"],
                                    "probabilities_before": before["probabilities"], "probabilities_after": row["probabilities"],
                                    "source": relative, "line": row["sequence"]+1})
                        if task == "classification":
                            slow = max(records, key=lambda row: row["elapsed_ms"])
                            meta = slow["response"]["meta"]
                            slow_blocks.append({"profile": profile, "n": n, "repeat": repeat,
                                "over_1s": sum(x["elapsed_ms"] > 1000 for x in records),
                                "over_5s": sum(x["elapsed_ms"] > 5000 for x in records),
                                "max_elapsed_ms": slow["elapsed_ms"], "sequence_at_max": slow["sequence"],
                                "native_compute_ms_at_max": meta["compute_ms"], "native_queue_ms_at_max": meta["queue_ms"],
                                "started_seconds_at_max": slow["started_seconds"], "source": relative})
        result["models"][model] = {"unique_changed_inputs": len(changes), "flip_events_vs_own_n1": sum(map(len, changes.values())),
            "changes": [{"task": key[0], "case_id": key[1], "fragment_id": key[2], "events": values}
                        for key, values in changes.items()], "classification_tails": slow_blocks}
    result.update(input_audit_sha256=sha(args.audit), source_sha256=sources,
                  script_sha256=sha(Path(__file__)), at_utc=datetime.now(timezone.utc).isoformat(),
                  inference_calls=0, root_cause_limit="native compute stage observed; JIT/graph/kernel substage attribution not proven")
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({m: {k: v[k] for k in ("unique_changed_inputs", "flip_events_vs_own_n1")}
                      for m, v in result["models"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
