"""Verify deployment boundaries and small Chinese examples, not model quality."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument("--target", action="append", required=True, help="model=http://host:port")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
key = os.environ["DECISION_EVAL_API_KEY"]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
questions = {
    "route": {"type": "choice", "instructions": "请选择处理这条记录的部门。", "criteria": {"returns": "退换货、破损商品", "billing": "账单、扣款、发票", "technical": "软件故障、服务中断"}},
    "receipt": {"type": "noul", "instructions": "这条记录是否明确说顾客提供了购物小票或付款凭证？"},
    "urgency": {"type": "score", "instructions": "根据记录中明确写出的处理时限判断紧急程度。", "criteria": ["下个月前处理即可", "本周内处理", "今天必须处理"]},
}
cases = [
    ("returns", True, 2, "客服记录：顾客昨天下单，今天收到的商品破损。顾客提供了购物小票，要求今天完成换货。"),
    ("billing", False, 0, "记录：客户只询问下一次账单何时扣款。客户没有提供付款凭证；这不是故障投诉。客户明确说下个月前处理即可。"),
    ("technical", False, 2, "记录：网站登录页持续返回服务器错误，所有用户都无法登录。用户没有收据，也没有退款诉求；必须今天恢复服务。"),
]


def request(base, path, body=None, credential=key):
    headers = {"Content-Type": "application/json"}
    if credential:
        headers["Authorization"] = "Bearer " + credential
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    started = time.perf_counter()
    try:
        with opener.open(urllib.request.Request(base + path, data=payload, headers=headers), timeout=120) as response:
            status, data = response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        status, data = exc.code, exc.read().decode()
    return status, data, round((time.perf_counter() - started) * 1000, 3)


results = []
for target in args.target:
    model, base = target.split("=", 1)
    base = base.rstrip("/")
    security = []
    for credential, expected in [(None, 401), ("deliberately-wrong", 401), (key, 200)]:
        status, data, elapsed = request(base, "/health", credential=credential)
        security.append({"case": "health_missing_wrong_valid_key", "status": status, "passed": status == expected})
        time.sleep(0.3)
    body = {"model": model, "state": cases[0][3], "questions": questions}
    for altered, expected in [({**body, "model": "wrong-model"}, 422), ({**body, "images": ["file:///not-permitted"]}, 422), ({**body, "state": "x" * 70000}, 413)]:
        status, data, elapsed = request(base, "/v1/systemone", altered)
        security.append({"case": "invalid_model_media_oversized", "status": status, "passed": status == expected})
        time.sleep(0.3)
    status, data, elapsed = request(base, "/v1/systemone", {**body, "state": "token " * 5000})
    rejected = status == 422
    if status == 200 and isinstance(data, dict):
        errors = data.get("answers", {})
        rejected = set(errors) == set(questions) and all(answer.get("error") == "max_length_exceeded" for answer in errors.values())
    security.append({"case": "token_overflow_not_silently_truncated", "status": status, "passed": rejected})
    time.sleep(0.3)
    rows = []
    for index in range(13):
        route, receipt, urgency, state = cases[index] if index < len(cases) else cases[0]
        body = {"model": model, "state": state, "questions": questions}
        status, data, elapsed = request(base, "/v1/systemone", body)
        valid = status == 200
        matches = {}
        if valid:
            answers = data.get("answers", {})
            valid = set(answers) == set(questions) and all(not answer.get("error") for answer in answers.values())
            for qid in ["route", "urgency"]:
                probabilities = answers.get(qid, {}).get("probabilities", {})
                valid = valid and bool(probabilities) and all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()) and abs(sum(probabilities.values()) - 1) < 0.005
            if valid:
                noul = answers["receipt"].get("noul")
                valid = isinstance(noul, (int, float)) and math.isfinite(noul) and 0 <= noul <= 1
            if valid:
                matches = {"route": answers["route"]["choice"] == route, "receipt": (noul >= 0.5) == receipt, "urgency": max(answers["urgency"]["probabilities"], key=answers["urgency"]["probabilities"].get) == str(urgency)}
        rows.append({"case": min(index, 3), "status": status, "valid": valid, "elapsed_ms": elapsed, "expected_label_matches": matches, "response": data})
        print(model, index, status, "valid=" + str(valid), elapsed, "ms", matches, flush=True)
        time.sleep(0.3)
    warm = [row["elapsed_ms"] for row in rows[3:] if row["valid"]]
    server_warm = [row["response"]["latency_ms"] for row in rows[3:] if row["valid"]]
    result = {"model": model, "endpoint": base, "security": security, "rows": rows, "small_smoke_median_client_ms": statistics.median(warm) if warm else None, "small_smoke_median_server_ms": statistics.median(server_warm) if server_warm else None, "deployment_checks_passed": all(row["valid"] for row in rows) and all(row["passed"] for row in security)}
    results.append(result)
simultaneous = []
if len(results) > 1:
    def simultaneous_request(target):
        body = {"model": target["model"], "state": cases[0][3], "questions": questions}
        status, data, elapsed = request(target["endpoint"], "/v1/systemone", body)
        answers = data.get("answers", {}) if isinstance(data, dict) else {}
        valid = status == 200 and set(answers) == set(questions) and all(not answer.get("error") for answer in answers.values())
        return {"model": target["model"], "status": status, "valid": valid, "elapsed_ms": elapsed, "response": data}

    with ThreadPoolExecutor(max_workers=len(results)) as pool:
        simultaneous = list(pool.map(simultaneous_request, results))
    print("Simultaneous model requests:", [(row["model"], row["status"], row["valid"]) for row in simultaneous], flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
source = Path(__file__).parent
identity = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ["server.py", "compose.yml", "nginx.conf", "Dockerfile", "smoke_test.py"]}
args.output.write_text(json.dumps({"tested_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": identity, "warning": "Three synthetic Chinese examples with repeats; not an accuracy or production benchmark", "models": results, "simultaneous_model_requests": simultaneous}, ensure_ascii=False, indent=2) + "\n")
raise SystemExit(0 if all(result["deployment_checks_passed"] for result in results) and all(row["valid"] for row in simultaneous) else 1)
