"""Independent pointwise records using reviewed official decision primitives."""

from __future__ import annotations

import math
import time


def score_answer(answer, question):
    probabilities = answer.get("probabilities", {})
    if question["type"] == "choice":
        keys = list(question["criteria"])
        if (answer.get("type") != "choice" or answer.get("error") or set(probabilities) != set(keys)
                or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                       for p in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > 0.005):
            raise RuntimeError("Invalid official choice distribution")
        return float(probabilities["relevant"])

    keys = [str(i) for i in range(len(question["criteria"]))]
    if (answer.get("type") != "score" or answer.get("error") or set(probabilities) != set(keys)
            or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values())
            or abs(sum(probabilities.values()) - 1) > 0.005):
        raise RuntimeError("Invalid official score distribution")
    return sum(i * probabilities[key] for i, key in enumerate(keys)) / (len(keys) - 1)


class DecisionReranker:
    def __init__(self, backend, runtime, max_length, validate_laya=None):
        self.backend, self.runtime = backend, runtime
        self.max_length, self.validate_laya = max_length, validate_laya

    def prepare(self, body):
        questions = body.get("questions")
        if not questions:
            raise ValueError("A frozen relevance question is required")
        if self.backend == "intern" and body["batch_size"] != 1:
            raise ValueError("Intern cross-state native batching is not validated; use batch 1")
        records, lengths = [], []
        for passage in body["documents"]:
            record = {"model": body["model"], "state": {"query": body["query"], "passage": passage},
                      "questions": questions}
            if self.backend == "d3":
                prepared = self.runtime.prepare(record["state"], questions)
                if prepared.errors or prepared.runnable != ["relevance"]:
                    raise ValueError("Official D3 encoder rejected the record")
                length = len(prepared.sequences["relevance"])
                records.append(prepared)
            elif self.backend == "clef":
                from joint_schema_model import encode_record

                _, processor = self.runtime
                encoded = encode_record(processor.tokenizer, record, max_length=self.max_length + 1,
                                        processor=processor)
                length = len(encoded.input_ids)
                records.append(encoded)
            elif self.backend == "laya":
                length = sum(self.validate_laya(self.runtime, record, self.max_length))
                records.append(record["state"])
            elif self.backend == "intern":
                _, encoded, _ = self.runtime.backend.encode(record)
                length = encoded["input_ids"].shape[-1]
                records.append(record)
            else:
                raise ValueError("Unsupported decision reranker")
            if length > self.max_length:
                raise ValueError("Input exceeds the evaluation token limit")
            lengths.append(length)
        return records, lengths

    def forward(self, records, questions):
        if self.backend == "d3":
            from d3_runtime import product_answer

            probabilities = self.runtime.probabilities(
                [r.sequences["relevance"] for r in records],
                [len(r.questions["relevance"].keys) for r in records])
            return [product_answer(r.questions["relevance"], p) for r, p in zip(records, probabilities)]
        if self.backend == "clef":
            from joint_schema_model import collate_records, systemone_answer

            model, processor = self.runtime
            logits = model(collate_records(records, processor.tokenizer.pad_token_id, next(model.parameters()).device))
            return [systemone_answer(questions["relevance"],
                    dict(zip(r.questions[0].option_ids, output[0].float().softmax(-1).tolist())))
                    for r, output in zip(records, logits)]
        if self.backend == "laya":
            results = self.runtime.predict_batch(records, questions, batch_size=len(records),
                                                max_len=self.max_length, head_max_len=self.max_length,
                                                sort_by_length=False)
            if any(r.get("usage", {}).get("truncated") or r.get("usage", {}).get("truncated_questions") for r in results):
                raise RuntimeError("Official Laya runtime truncated an accepted record")
            return [r["answers"]["relevance"] for r in results]
        return [self.runtime.predict(r)["answers"]["relevance"] for r in records]

    def __call__(self, body):
        import torch

        # Validate every record before any forward pass; no partial-pool success.
        records, lengths = self.prepare(body)
        batch_size = body["batch_size"]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        target = (self.runtime.backbone if self.backend == "d3" else self.runtime[0]
                  if self.backend == "clef" else self.runtime.backend.model
                  if self.backend == "intern" else self.runtime.model)
        observed = []

        def observe(module, args, kwargs):
            inputs = kwargs.get("input_ids", args[0] if args else None)
            if isinstance(inputs, dict):
                inputs = inputs["input_ids"]
            if inputs is None or inputs.device.type != "cuda":
                raise RuntimeError("Non-GPU forward is forbidden")
            observed.append({"rows": inputs.shape[0], "tokens": inputs.shape[1],
                             "device": str(inputs.device), "parameter_dtype": str(next(module.parameters()).dtype),
                             "autocast": torch.is_autocast_enabled("cuda")})

        hook = target.register_forward_pre_hook(observe, with_kwargs=True)
        scores, raw_answers, gpu_ms = [], [], 0.0
        try:
            with torch.inference_mode():
                for start in range(0, len(records), batch_size):
                    chunk = records[start:start + batch_size]
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    answers = self.forward(chunk, body["questions"])
                    torch.cuda.synchronize()
                    gpu_ms += (time.perf_counter() - started) * 1000
                    if len(answers) != len(chunk):
                        raise RuntimeError("Incomplete official batch")
                    scores.extend(score_answer(a, body["questions"]["relevance"]) for a in answers)
                    raw_answers.extend(answers)
        finally:
            hook.remove()
        if [r["rows"] for r in observed] != [min(batch_size, len(records) - i)
                                               for i in range(0, len(records), batch_size)]:
            raise RuntimeError("Official runtime did not execute the expected independent batches")
        return {
            "model": body["model"],
            "results": [{"index": i, "relevance_score": s, "answer": a}
                        for i, (s, a) in enumerate(zip(scores, raw_answers))],
            "usage": {"input_tokens": sum(lengths), "output_tokens": 0},
            "execution": {
                "kind": "independent_gpu_batches" if self.backend != "intern" else "independent_serial",
                "batch_size": batch_size, "forward_calls": len(observed), "batch_lengths": [r["rows"] for r in observed],
                "observed_forwards": observed,
                "input_tokens_per_document": lengths, "padded_tokens": sum(r["rows"] * r["tokens"] for r in observed),
                "gpu_inference_ms": round(gpu_ms, 3),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "backend": self.backend, "device": "cuda:0", "truncated": False, "cpu_fallback": False,
                "precision": "official_release", "score_mapping": (
                    "probability of criterion 'relevant'" if body["questions"]["relevance"]["type"] == "choice"
                    else "expected ordinal grade / maximum grade"
                ),
                "permutation_average": False,
            },
        }
