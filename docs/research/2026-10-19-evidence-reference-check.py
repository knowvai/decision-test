"""Official D3 reference diagnosis; immutable inputs and no network inference."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time


SEED = "evidence-reference-order-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open("rb") as stream:
        result = []
        for line in stream:
            assert line.endswith(b"\n"), "Incomplete JSONL record"
            result.append(json.loads(line))
        return result


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def controls():
    examples = [
        ("青禾商店退货期限和运费规则是什么？", "青禾商店允许七天内退货，运费由商店承担。", "青禾商店允许七天内退货。", "青禾商店销售文具，也提供客户咨询入口。", "蓝湾商店允许七天内退货，运费由商店承担。"),
        ("松桥展馆周六几点开门，门票多少钱？", "松桥展馆周六九点开门，门票二十元。", "松桥展馆的门票为二十元。", "松桥展馆的展品包括绘画和雕塑。", "北岭展馆周六九点开门，门票二十元。"),
        ("远帆设备保修多久，怎样申请？", "远帆设备保修两年，申请时向客服提交购买凭证。", "远帆设备的保修期为两年。", "远帆设备是一款便携式设备，包装采用纸盒。", "晨星设备保修两年，申请时向客服提交购买凭证。"),
        ("银杉项目会议在哪天举行，地点在哪里？", "银杉项目会议周三举行，地点是三楼第二会议室。", "银杉项目会议定于周三举行。", "银杉项目涉及文档整理，团队正在准备资料。", "白杨项目会议周三举行，地点是三楼第二会议室。"),
        ("星湾文件系统如何导出文件，支持什么格式？", "星湾文件系统在文件菜单中点击导出，支持PDF格式。", "星湾文件系统支持导出PDF格式。", "星湾文件系统提供文件浏览和收藏功能。", "月溪文件系统在文件菜单中点击导出，支持PDF格式。"),
        ("石岚班车几点发车，从哪个站出发？", "石岚班车每天八点发车，从南门站出发。", "石岚班车从南门站出发。", "石岚班车的车身是白色，座椅为蓝色。", "云港班车每天八点发车，从南门站出发。"),
    ]
    result = []
    for index, (query, *documents) in enumerate(examples):
        for kind, document in zip(("complete", "partial", "topic_only", "wrong_object"), documents):
            result.append({"case_id": f"synthetic:{index}", "fragment_id": kind,
                           "state": {"query": query, "document": document},
                           "gold": "relevant" if kind in ("complete", "partial") else "irrelevant"})
    return result


def select_reverse(jobs):
    return sorted(jobs, key=lambda j: digest([bytes.fromhex("6b6e6f77762d65766964656e63652d7265666572656e63652d6f726465722d7631").decode("utf-8"), j["case_id"], j["fragment_id"]]))[:128]


def key(job):
    return job["case_id"], job["fragment_id"]


def reverse(question):
    result = deepcopy(question)
    criteria = result["relevance"]["criteria"]
    result["relevance"]["criteria"] = dict(reversed(list(criteria.items())))
    return result


def request_digest(state, question):
    # Criteria insertion order changes answer-code assignment despite JSON object equivalence.
    return digest([state, question, list(question["relevance"]["criteria"].items())])


def validate_answer(raw, tokens):
    answer = raw["answers"]["relevance"]
    assert "error" not in answer
    probabilities = answer["probabilities"]
    assert set(probabilities) == {"relevant", "irrelevant"}
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1) < 1e-5
    assert answer["choice"] in probabilities
    assert probabilities[answer["choice"]] == max(probabilities.values())
    assert raw["usage"]["input_tokens"] == tokens and raw["usage"]["output_tokens"] == 0
    return answer["choice"], probabilities


def validate_encoding(text, worker_text, official_ids, worker_ids, max_length):
    assert text == worker_text
    assert official_ids and worker_ids == official_ids
    # The official runtime uses None for an unset package context limit.
    if max_length is not None:
        assert isinstance(max_length, int) and not isinstance(max_length, bool) and max_length > 0
        assert len(official_ids) <= max_length, "Full input exceeds runtime limit; no truncation"


def infer(args):
    import torch
    import transformers
    from transformers import AutoModel
    from vllm_srun.families.decision3 import prompt

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    jobs = rows(args.data / "evidence-holdout.private.jsonl")
    assert len(jobs) == len({key(j) for j in jobs}) == 643
    question = read(args.data / "protocol.json")["variants"]["evidence"]["baseline"]
    old = rows(args.old)
    assert [key(j) for j in jobs] == [key(r) for r in old]
    assert all(r["status"] == "valid" and r["http_status"] == 200 for r in old)
    packages = read(args.lock)["models"]
    package = next(p for p in packages if p["directory"] == args.model)
    for name, expected in package["files_sha256"].items():
        assert sha(args.package / name) == expected, name
    source_files = [Path(__file__), args.lock, args.old, args.data / "protocol.json",
                    args.data / "evidence-holdout.private.jsonl",
                    args.data / "evidence-holdout.labels.private.jsonl"]
    hashes = {str(p): sha(p) for p in source_files}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert torch.cuda.is_available()
    save(output / "protocol.json", {"model": args.model, "revision": package["official_revision"],
         "source_sha256": hashes, "question": question, "reverse_ids": [key(j) for j in select_reverse(jobs)],
         "synthetic_sha256": digest(controls()), "planned": 795, "retries": 0,
         "torch": torch.__version__, "transformers": transformers.__version__,
         "gpu": torch.cuda.get_device_name(0), "at_utc": datetime.now(timezone.utc).isoformat()})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Reference deadline")))
    signal.alarm(2400)
    began = time.monotonic()
    model = AutoModel.from_pretrained(str(args.package), trust_remote_code=True,
            local_files_only=True, device="cuda:0", batch_size=1, verify="full")
    signal.alarm(0)
    runtime = model.runtime
    assert str(runtime.device) == "cuda:0" and str(runtime.readout.dtype) == "torch.float32"
    assert runtime.attention_mode == "noncausal_full_attention" and not runtime.permutation_average
    worker_tokenizer, _ = prompt.load_tokenizer(args.package)
    save(output / "loaded.json", {"seconds": time.monotonic() - began, "kernels": runtime.kernels,
         "attention_mode": runtime.attention_mode, "readout_dtype": str(runtime.readout.dtype),
         "backbone_dtype": str(next(runtime.backbone.parameters()).dtype),
         "fast_enabled": runtime.fast is not None, "fast_skipped": runtime.fast_skipped,
         "max_input_tokens": runtime.max_length})
    captured = []

    def capture(_module, _args, kwargs):
        captured.append(kwargs["input_ids"].detach().cpu().tolist())

    hook = runtime.backbone.register_forward_pre_hook(capture, with_kwargs=True)
    attempted = valid = 0
    stage = "not_started"
    try:
        with (output / "requests.jsonl").open("x") as stream, (output / "intents.jsonl").open("x") as intents:
            for stage, selected, q in (("baseline", jobs, question), ("reversed", select_reverse(jobs), reverse(question)),
                                        ("synthetic", controls(), question)):
                for job in selected:
                    text = runtime.text(job["state"], q["relevance"])
                    official_ids = runtime.tokenizer(text, add_special_tokens=False)["input_ids"]
                    worker_text = prompt.render(prompt.user_prompt(job["state"], q["relevance"]["instructions"],
                              prompt.option_texts("choice", q["relevance"]["criteria"]), runtime.codes[:2]))
                    validate_encoding(text, worker_text, official_ids,
                                      worker_tokenizer.encode(worker_text, add_special_tokens=False).ids,
                                      runtime.max_length)
                    identity = {"stage": stage, "case_id": job["case_id"], "fragment_id": job["fragment_id"],
                                "input_sha256": digest(job["state"]), "request_sha256": request_digest(job["state"], q),
                                "tokens": len(official_ids), "token_ids_sha256": digest(official_ids), "ordinal": attempted + 1}
                    intents.write(json.dumps(identity) + "\n")
                    intents.flush()
                    attempted += 1
                    captured.clear()
                    start = time.monotonic()
                    signal.alarm(300)
                    raw = model.system_one(state=job["state"], questions=q)
                    runtime.synchronize()
                    signal.alarm(0)
                    elapsed = time.monotonic() - start
                    prediction, probabilities = validate_answer(raw, len(official_ids))
                    assert captured == [[official_ids]], "Actual backbone tokens differ"
                    record = {**identity, "status": "valid", "calls": 1, "prediction": prediction,
                              "probabilities": probabilities, "elapsed_seconds": elapsed,
                              "actual_token_ids_match": True}
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    valid += 1
                    if valid % 50 == 0:
                        print(json.dumps({"model": args.model, "stage": stage, "valid": valid}), flush=True)
        assert valid == attempted == 795
        assert all(sha(p) == expected for p, expected in hashes.items())
        save(output / "complete.json", {"attempted": attempted, "valid": valid, "failed": 0,
             "not_attempted": 0, "source_unchanged": True, "requests_sha256": sha(output / "requests.jsonl")})
    except BaseException as exc:
        save(output / "failed.json", {"stage": stage, "attempted": attempted, "valid": valid,
             "failed_or_unknown": attempted-valid, "not_attempted": 795-attempted, "error_type": type(exc).__name__})
        raise
    finally:
        signal.alarm(0)
        hook.remove()


def confusion(jobs, records):
    assert [key(j) for j in jobs] == [key(r) for r in records]
    result = Counter()
    for job, record in zip(jobs, records):
        y, p = job["gold"] == "relevant", record["prediction"] == "relevant"
        result["tp" if y and p else "fn" if y else "fp" if p else "tn"] += 1
    return dict(result)


def compare(a, b):
    assert [key(r) for r in a] == [key(r) for r in b]
    delta = [abs(x["probabilities"]["relevant"] - y["probabilities"]["relevant"]) for x, y in zip(a, b)]
    changes = [list(key(x)) for x, y in zip(a, b) if x["prediction"] != y["prediction"]]
    return {"n": len(a), "changed": len(changes), "changed_ids": changes,
            "max_probability_delta": max(delta), "mean_probability_delta": sum(delta)/len(delta),
            "probability_delta_gt_0_01": sum(d > .01 for d in delta)}


def analyze(args):
    def offline(event, _args):
        if event in {"socket.connect", "subprocess.Popen", "os.system"}:
            raise RuntimeError("Analysis is offline")

    sys.addaudithook(offline)
    jobs = rows(args.data / "evidence-holdout.private.jsonl")
    labels = {r["case_id"]: r for r in rows(args.data / "evidence-holdout.labels.private.jsonl")}
    results = {}
    for model in ("d3-mini", "d3-flash"):
        path = args.root / model
        complete, protocol = read(path / "complete.json"), read(path / "protocol.json")
        assert complete["valid"] == complete["attempted"] == protocol["planned"] == 795
        assert complete["requests_sha256"] == sha(path / "requests.jsonl")
        records, intents = rows(path / "requests.jsonl"), rows(path / "intents.jsonl")
        assert len(records) == len(intents) == 795
        for i, (record, intent) in enumerate(zip(records, intents), 1):
            assert record["ordinal"] == i and record["status"] == "valid" and record["calls"] == 1
            assert all(record[k] == v for k, v in intent.items()) and record["actual_token_ids_match"]
        baseline = [r for r in records if r["stage"] == "baseline"]
        rev = [r for r in records if r["stage"] == "reversed"]
        synth = [r for r in records if r["stage"] == "synthetic"]
        old = rows(args.prior / model / "evidence-holdout-baseline.jsonl")
        for local in (args.data / "protocol.json", args.data / "evidence-holdout.private.jsonl",
                      args.data / "evidence-holdout.labels.private.jsonl",
                      args.prior / model / "evidence-holdout-baseline.jsonl"):
            expected = [v for p, v in protocol["source_sha256"].items() if Path(p).name == local.name]
            assert expected == [sha(local)]
        by_key = {key(r): r for r in baseline}
        reverse_jobs = select_reverse(jobs)
        for stage, js, rs, q in (("baseline", jobs, baseline, protocol["question"]),
                                 ("reversed", reverse_jobs, rev, reverse(protocol["question"])),
                                 ("synthetic", controls(), synth, protocol["question"])):
            assert [key(j) for j in js] == [key(r) for r in rs]
            for j, r in zip(js, rs):
                assert r["input_sha256"] == digest(j["state"])
                assert r["request_sha256"] == request_digest(j["state"], q)
        results[model] = {"historical_worker": confusion(jobs, old), "official_reference": confusion(jobs, baseline),
            "worker_reference": compare(old, baseline), "reverse_order": compare([by_key[key(j)] for j in reverse_jobs], rev),
            "reverse_original_counts": confusion(reverse_jobs, [by_key[key(j)] for j in reverse_jobs]),
            "reverse_counts": confusion(reverse_jobs, rev), "synthetic_counts": confusion(controls(), synth),
            "synthetic_errors": [list(key(j)) for j, r in zip(controls(), synth) if j["gold"] != r["prediction"]],
            "grades": {str(g): {"n": sum(labels[j["case_id"]]["grades"][j["fragment_id"]] == g for j in jobs),
                "reference_kept": sum(labels[j["case_id"]]["grades"][j["fragment_id"]] == g and r["prediction"] == "relevant" for j, r in zip(jobs, baseline))} for g in range(4)},
            "artifact_sha256": {name: sha(path / name) for name in ("protocol.json", "loaded.json", "requests.jsonl", "intents.jsonl", "complete.json")}}
    save(args.output, {"models": results, "valid": 1590, "passed": True, "new_human_labels": 0,
         "limits": ["Historical worker versus new reference, not simultaneous latency A/B", "Original proxy labels unchanged", "No Clef reference inference", "Synthetic controls are not population accuracy"]})
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("infer")
    for name in ("data", "old", "package", "lock", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--model", required=True, choices=("d3-mini", "d3-flash"))
    p = subs.add_parser("analyze")
    for name in ("data", "prior", "root", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    (infer if args.action == "infer" else analyze)(args)
