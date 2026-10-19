"""CPU-only native encoding gate in the retained owned SGLang container."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys


ENCODE = r'''
import hashlib
import inspect
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer
from sglang.srt.entrypoints.systemone.joint_schema import encode_joint_schema, SystemOneRequest

payload = json.load(sys.stdin)
tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
rows = []
for item in payload["items"]:
    for mode, questions in payload["questions"].items():
        request = SystemOneRequest(model=payload["served_model"], state=item["state"], questions=questions)
        ids, layout = encode_joint_schema(tokenizer, request, max_length=1000000, image_token_counts=[])
        count = len(ids)
        within_limit = count <= 8192
        try:
            actual_ids, _ = encode_joint_schema(tokenizer, request, max_length=8192, image_token_counts=[])
            assert within_limit and actual_ids == ids
        except ValueError:
            assert not within_limit
        rows.append({**{k: v for k, v in item.items() if k != "state"}, "mode": mode,
                     "tokens": count, "within_limit": within_limit})
source = Path(inspect.getfile(encode_joint_schema))
print(json.dumps({"rows": rows, "encoder_path": str(source),
                  "encoder_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "tokenizer_sha256": hashlib.sha256(Path("/model/tokenizer.json").read_bytes()).hexdigest()}))
'''


def main(args):
    sys.path.insert(0, str(args.root / "source-r1/evaluation/runner/scripts"))
    import decision_evidence_benchmark as e
    data, _ = e.load_dataset(args.root / "dataset-v1")
    def identity():
        template = ('{"id":{{json .Id}},"image":{{json .Image}},"pid":{{json .State.Pid}},'
                    '"started_at":{{json .State.StartedAt}},"running":{{json .State.Running}},'
                    '"owned":{{json (index .Config.Labels "evaluation.decision.acceptance")}}}')
        return json.loads(subprocess.check_output(["docker", "inspect", args.container, "--format", template], text=True))
    before = identity()
    assert before["owned"] == "true" and before["running"]
    items = []
    for phase, (pools, _) in data.items():
        for pool in pools:
            for candidate in pool["candidates"]:
                state = e.p.pair_state(pool, candidate)
                items.append({"phase": phase, "case_id": pool["case_id"],
                              "fragment_id": candidate["fragment_id"],
                              "input_sha256": e.b.digest(state), "state": state})
    payload = {"served_model": args.model, "items": items,
               "questions": {mode: e.questions(mode) for mode in ("choice", "noul")}}
    result = subprocess.run(["docker", "exec", "-i", "-e", "CUDA_VISIBLE_DEVICES=",
                             "-e", "HF_HUB_OFFLINE=1", "-e", "TOKENIZERS_PARALLELISM=false",
                             args.container, "python3", "-c", ENCODE],
                            input=json.dumps(payload), text=True, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError("Native CPU encoder failed; stderr deliberately not echoed")
    encoded = json.loads(result.stdout)
    assert identity() == before
    groups = {}
    for phase in data:
        rows = [r for r in encoded["rows"] if r["phase"] == phase]
        rejected = [r for r in rows if not r["within_limit"]]
        groups[phase] = {"pairs": len(rows) // 2, "max_tokens": max(r["tokens"] for r in rows),
                         "over_limit_pairs": len({(r["case_id"], r["fragment_id"]) for r in rejected}),
                         "over_limit_pools": len({r["case_id"] for r in rejected})}
    report = {"at": e.p.now(), "model": args.model, "container": before,
              "dataset_protocol_sha256": e.sha(args.root / "dataset-v1/protocol.json"),
              "script_sha256": e.sha(__file__), "context_limit": 8192,
              "method": "Exact deployed native joint-schema encoder and tokenizer; CPU only; no weights or inference",
              "inference_calls": 0, "truncations": 0, "scope": "Retained SGLang model only, not all-model readiness",
              "passed": all(r["within_limit"] for r in encoded["rows"]),
              "splits": groups, **encoded}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("model", "passed", "inference_calls", "splits")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
