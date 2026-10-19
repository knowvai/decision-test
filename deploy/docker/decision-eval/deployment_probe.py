"""Small synthetic native-runtime acceptance; not the public benchmark runner."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from launch_reviewed import MODELS, read_key


def valid_probabilities(probabilities, keys):
    return (isinstance(probabilities, dict) and set(probabilities) == set(keys)
            and all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                    for p in probabilities.values())
            and abs(sum(probabilities.values()) - 1) < 0.01)


def native_choice_request(family, payload):
    """Map the common choice contract without changing the native inference path."""
    assert next(iter(payload["questions"].values()))["type"] == "choice"
    return native_question_request(family, payload)


def native_question_request(family, payload):
    """Share transport and complete-input guards for native choice and noul."""
    assert len(payload["questions"]) == 1
    question = next(iter(payload["questions"].values()))
    assert question["type"] in ("choice", "noul")
    if question["type"] == "noul":
        assert "criteria" not in question
    if family == "jev":
        state = payload["state"]
        return "/v1/decide", {
            "kind": question["type"], "state": state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
            "question": question["instructions"],
            **({"options": list(question["criteria"].values())} if question["type"] == "choice" else {}),
            "strategy": "single", "thinking": "off",
        }
    if family == "laya":
        payload = {**payload, "max_len": 8192, "head_max_len": 8192}
    if family == "worker":
        payload = {**payload, "questions": {
            name: {**value, "require_full_input": True} for name, value in payload["questions"].items()}}
    return "/v1/systemone", payload


def native_choice_result(family, payload, response):
    name, question = next(iter(payload["questions"].items()))
    criteria = question["criteria"]
    if family == "jev":
        assert response["options"] == list(criteria.values())
        assert len(response["probabilities"]) == len(criteria)
        probabilities = dict(zip(criteria, response["probabilities"]))
        index = response["choice_index"]
        assert type(index) is int and 0 <= index < len(criteria)
        choice = list(criteria)[index]
        assert response["choice"] == criteria[choice]
    else:
        answer = response["answers"][name]
        assert answer["type"] == "choice"
        probabilities, choice = answer["probabilities"], answer["choice"]
        if payload["model"].startswith("Vela-"):
            assert answer.get("input_coverage") == "complete"
    assert choice in criteria and valid_probabilities(probabilities, criteria)
    assert not response.get("usage", {}).get("truncated", False)
    assert not response.get("execution", {}).get("truncated", False)
    assert not response.get("execution", {}).get("cpu_fallback", False)
    return choice, probabilities


def native_noul_result(family, payload, response):
    """Decode native true probability, never fit a retention threshold."""
    name, question = next(iter(payload["questions"].items()))
    assert question["type"] == "noul"
    assert not response.get("usage", {}).get("truncated", False)
    assert not response.get("usage", {}).get("truncated_tokens", 0)
    assert not response.get("execution", {}).get("truncated", False)
    assert not response.get("execution", {}).get("cpu_fallback", False)
    native_choice, value = None, None
    if family == "jev":
        assert response["kind"] == response["effective_kind"] == "noul"
        assert response["adaptation"] == "native"
        assert response["options"] == ["false", "true"]
        assert len(response["probabilities"]) == 2
        probabilities = dict(zip(response["options"], response["probabilities"]))
        index = response["choice_index"]
        assert type(index) is int and index in (0, 1)
        native_choice = response["choice"]
        assert native_choice == response["options"][index]
        assert response["num_model_requests"] == 1
        assert not response.get("thinking", {}).get("used", False)
        readout = "native_choice"
    else:
        answer = response["answers"][name]
        assert answer["type"] == "noul"
        if payload["model"].startswith("Vela-"):
            assert answer.get("input_coverage") == "complete"
        value = answer["noul"]
        assert type(value) in (bool, int, float) and math.isfinite(value) and 0 <= value <= 1
        probabilities = answer.get("probabilities")
        if probabilities is not None:
            assert valid_probabilities(probabilities, ("false", "true"))
            if type(value) is not bool:
                assert abs(probabilities["true"] - value) <= 0.0001
        else:
            probabilities = {"false": 1 - float(value), "true": float(value)}
        if type(value) is bool:
            native_choice = "true" if value else "false"
            readout = "native_boolean"
        else:
            readout = "fixed_binary_argmax_false_on_tie"
    assert valid_probabilities(probabilities, ("false", "true"))
    choice = native_choice or ("true" if probabilities["true"] > probabilities["false"] else "false")
    native = {k: response[k] for k in ("kind", "effective_kind", "adaptation", "options", "probabilities",
              "choice_index", "choice", "num_model_requests", "thinking") if k in response} if family == "jev" else {
                  "answers": {name: {k: answer[k] for k in ("type", "noul", "probabilities", "input_coverage") if k in answer}}}
    native.update(usage=response.get("usage") or {}, execution=response.get("execution") or {})
    return {"prediction": "relevant" if choice == "true" else "irrelevant",
            "noul": value, "native_choice": native_choice, "readout": readout,
            "probabilities": probabilities, "usage": response.get("usage"),
            "execution": response.get("execution"), "native_response": native}


def probe(args):
    root = args.root
    family, _, model = MODELS[args.model]
    key = read_key(args.env_file)
    base = f"http://127.0.0.1:{args.port}"
    output = root / "acceptance-20261015/evidence" / (args.model + ".probe.json")
    report = {"model": model, "at": datetime.now(timezone.utc).isoformat(),
              "scope": "synthetic runtime acceptance, not accuracy or optimized latency"}

    def call(path, body=None, credential=key, timeout=300, base_override=None):
        start = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if credential:
            headers["Authorization"] = "Bearer " + credential
        request = urllib.request.Request((base_override or base) + path, headers=headers,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode())
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                status, raw = response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read().decode()
        except (OSError, TimeoutError) as error:
            return {"status": 0, "error": type(error).__name__, "seconds": time.perf_counter() - start}
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        return {"status": status, "response": value, "seconds": time.perf_counter() - start}

    def save():
        data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        output.with_name(output.stem + "." + str(time.time_ns()) + ".json").write_text(data)
        output.write_text(data)

    auth_credentials = [None, "incorrect-evaluation-key"]
    if key.swapcase() != key:
        auth_credentials.append(key.swapcase())
    report["auth"] = [call("/health", credential=k) for k in auth_credentials]
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        ready = call("/health", timeout=5)
        if family == "neohorse" and ready["status"] == 200:
            deployment = json.loads(output.with_name(args.model + ".deployment.json").read_text())
            ready = call("/health", timeout=5,
                         base_override=f"http://127.0.0.1:{deployment['native_port'] - 1}")
        if ready["status"] == 200:
            break
        state = subprocess.check_output(["docker", "inspect", "decision-acceptance-model",
            "--format", "{{.State.Status}}"], text=True).strip()
        if state != "running" or "failed" in str(ready.get("response", {})).lower():
            break
        time.sleep(3)
    report["ready"] = ready
    if ready["status"] != 200:
        report["accepted"] = False
        save()
        print(json.dumps({"model": args.model, "ready": ready, "accepted": False}), flush=True)
        return False
    report["auth"] = [call("/health", credential=k) for k in auth_credentials]
    report["models"] = call("/v1/models")
    report["metrics_before"] = call("/metrics")

    def body(index, classification=False):
        state = f"记录编号 {index}。" + ("用户查询本月账单与扣款明细。" if index % 2 == 0 else "用户报告软件登录页面报错。")
        criteria = {"relevant": "与账单扣款问题相关", "irrelevant": "与账单扣款问题无关"}
        instructions = "这条记录与查询账单扣款明细的用户问题是否相关？"
        if classification:
            criteria = {"billing": "账单扣款明细", "software": "软件登录故障", **{f"other{i}": f"其他类别{i}" for i in range(16)}}
            instructions = "选择与这条记录最匹配的类别。"
        q = {"type": "choice", "instructions": instructions, "criteria": criteria}
        if family == "worker":
            q["require_full_input"] = True
        return {"model": model, "state": state, "questions": {"decision": q}}

    def decide(payload):
        q = payload["questions"]["decision"]
        if family == "jev":
            path, request = native_choice_request(family, payload)
            result = call(path, request)
            data = result.get("response", {})
            if result["status"] == 200:
                result["probabilities"] = dict(zip(q["criteria"], data.get("probabilities", [])))
        elif family == "llm":
            result = call("/v1/chat/completions", {"model": model, "temperature": 0, "max_tokens": 64,
                "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "structured_outputs": {"json": {"type": "object", "properties": {
                    "choice": {"type": "string", "enum": list(q["criteria"])}}, "required": ["choice"], "additionalProperties": False}}})
            if result["status"] == 200:
                try:
                    result["choice"] = json.loads(result["response"]["choices"][0]["message"]["content"])["choice"]
                except (KeyError, ValueError, TypeError):
                    pass
        else:
            path, request = native_choice_request(family, payload)
            result = call(path, request)
            data = result.get("response", {})
            if result["status"] == 200:
                result["probabilities"] = data.get("answers", {}).get("decision", {}).get("probabilities", {})
        probabilities = result.get("probabilities", {})
        result["valid"] = result["status"] == 200 and (
            result.get("choice") in q["criteria"] if family == "llm" else
            valid_probabilities(probabilities, q["criteria"]))
        if family == "laya" and result["status"] == 200:
            result["valid"] = result["valid"] and not result["response"].get("usage", {}).get("truncated", False)
        if args.model.startswith("vela-") and result["status"] == 200:
            result["valid"] = result["valid"] and result["response"].get("answers", {}).get("decision", {}).get("input_coverage") == "complete"
        return result

    if family == "reranker":
        docs = [body(i)["state"] for i in range(args.pool)]
        report["singles"] = [call("/rerank", {"model": model, "query": "查询账单扣款明细", "documents": [doc]}) for doc in docs[:4]]
        report["pool"] = call("/rerank", {"model": model, "query": "查询账单扣款明细", "documents": docs})
        data = report["pool"].get("response", {})
        rows = data.get("results", []) if isinstance(data, dict) else []
        report["valid_pool"] = report["pool"]["status"] == 200 and sorted(row["index"] for row in rows) == list(range(args.pool)) and all(math.isfinite(row["relevance_score"]) for row in rows)
        if report["valid_pool"] and all(row["status"] == 200 for row in report["singles"]):
            scores = {row["index"]: row["relevance_score"] for row in rows}
            report["max_probability_delta"] = max(abs(row["response"]["results"][0]["relevance_score"] - scores[i]) for i, row in enumerate(report["singles"]))
    elif family == "laya":
        report["classification"] = decide(body(0, classification=True))
        report["singles"] = [decide(body(i)) for i in range(4)]
        sample = body(0)
        report["pool"] = call("/v1/systemone/batch", {"model": model,
            "states": [body(i)["state"] for i in range(args.pool)], "questions": sample["questions"],
            "max_len": 8192, "head_max_len": 8192, "batch_size": 8})
        data = report["pool"].get("response", {})
        rows = data.get("results", []) if isinstance(data, dict) else []
        report["valid_pool"] = report["pool"]["status"] == 200 and len(rows) == args.pool and all(
            valid_probabilities(row.get("answers", {}).get("decision", {}).get("probabilities", {}), {"relevant", "irrelevant"})
            and not row.get("usage", {}).get("truncated", False) for row in rows)
        if report["valid_pool"] and all(row["valid"] for row in report["singles"]):
            report["max_probability_delta"] = max(abs(single["probabilities"][key] - repeated["answers"]["decision"]["probabilities"][key])
                for single, repeated in zip(report["singles"], rows) for key in single["probabilities"])
            report["batch_decision_flips"] = sum(
                max(single["probabilities"], key=single["probabilities"].get) !=
                max(repeated["answers"]["decision"]["probabilities"], key=repeated["answers"]["decision"]["probabilities"].get)
                for single, repeated in zip(report["singles"], rows))
        report["overlength"] = call("/v1/systemone", {**body(0), "state": "测试文本。" * 100,
                                                     "max_len": 16, "head_max_len": 8192})
    else:
        report["classification"] = decide(body(0, classification=True))
        report["singles"] = [decide(body(i)) for i in range(4)]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.pool) as pool:
            report["pool"] = list(pool.map(lambda i: decide(body(i)), range(args.pool)))
        report["pool_seconds"] = time.perf_counter() - start
        report["valid_pool"] = all(row["valid"] for row in report["pool"])
        if family != "llm" and report["valid_pool"] and all(row["valid"] for row in report["singles"]):
            report["max_probability_delta"] = max(abs(single["probabilities"][key] - repeated["probabilities"][key])
                for single, repeated in zip(report["singles"], report["pool"]) for key in single["probabilities"])
            report["batch_decision_flips"] = sum(max(single["probabilities"], key=single["probabilities"].get) != max(repeated["probabilities"], key=repeated["probabilities"].get)
                for single, repeated in zip(report["singles"], report["pool"]))
    if family in {"jev", "llm", "sglang"}:
        overlength = body(0)
        overlength["state"] = "测试输入。" * 6000
        report["overlength"] = decide(overlength)
    report["metrics_after"] = call("/metrics")
    report["valid_singles"] = all(row.get("valid", row["status"] == 200) for row in report["singles"])
    report["accepted"] = (all(row["status"] == 401 for row in report["auth"])
                          and report["valid_pool"] and report["valid_singles"]
                          and report.get("classification", {"valid": True})["valid"])
    if family == "laya":
        report["accepted"] = report["accepted"] and report["overlength"]["status"] == 422
    elif family in {"jev", "llm", "sglang"}:
        report["accepted"] = report["accepted"] and report["overlength"]["status"] in {400, 422}
    save()
    print(json.dumps({k:report[k] for k in ("model", "accepted", "valid_pool")}, ensure_ascii=False), flush=True)
    return report["accepted"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30889)
    parser.add_argument("--pool", type=int, default=49)
    parser.add_argument("--startup-timeout", type=int, default=2400)
    raise SystemExit(0 if probe(parser.parse_args()) else 1)
