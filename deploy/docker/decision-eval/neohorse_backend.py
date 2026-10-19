"""Batch the author's CPUHead records through the native vLLM pooling API."""

import sys
from pathlib import Path

import httpx


class NeoHorseBackend:
    def __init__(self, model_path, base_url, timeout=300):
        root = Path(model_path).resolve(strict=True)
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "infer" / "vllm"))
        from head import CPUHead

        self.head = CPUHead(str(root))
        self.url = base_url.rstrip("/") + "/pooling"
        self.health_url = base_url.rstrip("/") + "/health"
        self.timeout = timeout
        self.session = httpx.Client(timeout=timeout, trust_env=False)

    def is_ready(self):
        try:
            return self.session.get(self.health_url, timeout=min(3, self.timeout)).status_code == 200
        except httpx.HTTPError:
            return False

    def batch(self, bodies):
        groups = [self.head.prepare(body) for body in bodies]
        rows = [row for group in groups for row in group]
        if any("image" in row for row in rows):
            raise ValueError("Evaluation supports text only")
        response = self.session.post(self.url, json={
            "model": "NeoHorse-Jev-4B", "input": [row["ids"] for row in rows],
            "task": "token_embed", "use_activation": False,
            "encoding_format": "float", "add_special_tokens": False,
        }, timeout=self.timeout)
        response.raise_for_status()
        outputs = response.json()["data"]
        if len(outputs) != len(rows) or sorted(item["index"] for item in outputs) != list(range(len(rows))):
            raise RuntimeError("Native pooling returned incomplete or duplicate indices")
        hidden = [item["data"] for item in sorted(outputs, key=lambda item: item["index"])]
        results, offset = [], 0
        for group in groups:
            result = self.head.result(group, hidden[offset:offset + len(group)])
            result["execution"] = {"runtime": "vllm-pooling-author-cpuhead", "head_device": "cpu"}
            results.append(result)
            offset += len(group)
        return results

    def __call__(self, body):
        return self.batch([body])[0]
